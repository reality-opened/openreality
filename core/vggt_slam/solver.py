import numpy as np
import gtsam
import matplotlib.pyplot as plt
import torch
import time
import open3d as o3d
from termcolor import colored

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

from vggt_slam.slam_utils import compute_image_embeddings, Accumulator
from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.graph import PoseGraph
from vggt_slam.scale_solver import estimate_scale_pairwise
from vggt_slam.metric_anchor import MetricScaleAnchor, combine_scale
from vggt_slam.junction_weight import junction_covariance_multiplier
from vggt_slam.viewer import Viewer

DEBUG = False

# Minimum number of conf-masked overlap points required to estimate a cross-submap scale.
# Shared by all three mask tiers in `add_edge` so the fallback ladder cannot quietly end
# below the floor the first two tiers enforce.
MIN_SCALE_POINTS = 100

def debug_visualize(pcd1_points, pcd2_points):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)
    pcd1.paint_uniform_color([1, 0, 0])  # red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_points)
    pcd2.paint_uniform_color([0, 0, 1])  # blue

    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Pairwise Point Clouds")

class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        lc_thres: float = 0.80,
        vis_voxel_size: float = None,
        skip_viewer: bool = False,
        metric_scale_anchor: bool = False,          # EXP-12: per-submap metric-ratio scale anchor (default OFF)
        metric_anchor_model: str = "da2-metric-hypersim-large",
        metric_anchor_keyframes: int = 3,
        metric_anchor_gate: float = 0.5,
        metric_anchor_min_valid: int = 200,
        weighted_junctions: bool = False):          # scale junction factor covariances by measured quality (default OFF)

        self.init_conf_threshold = init_conf_threshold
        self.vis_voxel_size = vis_voxel_size

        if skip_viewer:
            self.viewer = None
        else:
            self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.lc_thres = lc_thres

        self.temp_count = 0
        self.vggt_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()

        self.clip_model = None
        self.clip_preprocess = None

        # Junction-confidence weighting (opt-in): scale each cross-submap junction
        # between-factor's covariance by measured junction quality (scale-estimate dispersion,
        # overlap conf-mask coverage, VGGT image_match_ratio at LC junctions). Low-confidence
        # junctions get looser factors so the graph — and any metric-anchor correction — can
        # override them. Mapping lives in ONE place: vggt_slam/junction_weight.py. Default OFF.
        self.weighted_junctions = weighted_junctions

        # EXP-12 metric scale anchor (opt-in). Constructing the anchor is cheap: the
        # metric-depth weights load lazily on first submap (flags-OFF never imports them).
        self.metric_scale_anchor = metric_scale_anchor
        self.metric_anchor = None
        if metric_scale_anchor:
            self.metric_anchor = MetricScaleAnchor(
                model_name=metric_anchor_model,
                keyframes=metric_anchor_keyframes,
                gate=metric_anchor_gate,
                min_valid=metric_anchor_min_valid,
            )

    def set_clip_model(self, clip_model, clip_preprocess):
        """Store CLIP model and preprocess on the solver instance."""
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

    def reset(self):
        """Reset SLAM state without reloading models."""
        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.graph = PoseGraph()
        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None
        self.temp_count = 0
        self.vggt_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.viewer is None:
            return
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_world_frame.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(points_colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points_in_world_frame = np.asarray(pcd.points, dtype=np.float32)
            points_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        self.viewer.server.scene.add_point_cloud(
            name="pcd_"+name,
            points=points_in_world_frame,
            colors=points_colors,
            point_size=point_size,
            point_shape="circle",
        )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        points_in_world_frame = submap.get_points_in_world_frame(self.graph)
        points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        if self.viewer is None:
            return
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world(self.graph)
        images = submap.get_all_frames()
        self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def tranform_submap_to_canonical(self, proj_mat_world_to_cam, world_points):
        P_first_cam = proj_mat_world_to_cam[0].copy()

        # Apply transformation to camera matrices such that the first camera matrix of the submap is [I | 0]
        proj_mat_world_to_cam = proj_mat_world_to_cam @ np.linalg.inv(P_first_cam)

        # Apply transformation to points such that the first camera matrix of the submap is [I | 0]
        h, w = world_points.shape[1:3]
        for i in range(len(proj_mat_world_to_cam)):
            points_in_cam = world_points[i,...]
            points_in_cam_h = np.hstack([points_in_cam.reshape(-1, 3), np.ones((points_in_cam.shape[0] * points_in_cam.shape[1], 1))])
            points_in_cam_h = (P_first_cam @ points_in_cam_h.T).T # TODO Dominic check if we want to use P_prior here
            points_in_cam = points_in_cam_h[:, :3] / points_in_cam_h[:, 3:]
            world_points[i] = points_in_cam.reshape(h, w, 3)
        
        return proj_mat_world_to_cam, world_points

    def _log_anchor_decision(self, submap_id_curr, current_submap, dec):
        """Print the metric-anchor gate decision and flag untrusted submaps for the export
        sidecar (EXP-12 §6 tier-3 rich-overlap disagreement)."""
        if dec["applied"]:
            msg = (f"[metric-anchor] submap {submap_id_curr}: {dec['reason']} -> "
                   f"scale {dec['scale_geom']:.4f} (geom) -> {dec['s_metric']:.4f} (anchor)")
            color = "yellow" if dec["degenerate"] else "green"
            if dec["degenerate"]:
                msg += "  [degenerate overlap: geometry suspect, preferring anchor]"
            print(colored(msg, color))
        else:
            print(colored(f"[metric-anchor] submap {submap_id_curr}: {dec['reason']} -> "
                          f"keeping geometric scale {dec['scale_geom']:.4f}", "cyan"))
        if dec["untrusted"]:
            current_submap.metric_anchor_untrusted = True

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False, image_match_ratio=None):
        assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
        scale_factor = 1.0
        current_submap = self.map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)
        if submap_id_prev is not None:
            overlapping_node_id_prev = submap_id_prev + frame_id_prev

            # Estimate scale factor between submaps.
            prior_submap = self.map.get_submap(submap_id_prev)

            current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)
            good_mask = (prior_conf > prior_submap.get_conf_threshold()) * (current_conf > prior_submap.get_conf_threshold())
            good_mask = good_mask.reshape(-1)

            if np.sum(good_mask) < MIN_SCALE_POINTS:
                print(colored("Not enough overlapping points to estimate scale factor, using a less restrictive mask", 'red'))
                good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
                if np.sum(good_mask) < MIN_SCALE_POINTS: # Handle the case where loop closure frames do not have enough points.
                    good_mask = (prior_conf > 0).reshape(-1)
                    # The loosest tier needs the same floor as the two above it. Without
                    # it a handful of points (or none) reached estimate_scale_pairwise,
                    # whose median is then pure noise or NaN — and that scale is baked
                    # into the SL(4) edge for good. If even `conf > 0` cannot muster the
                    # floor, the two submaps do not meaningfully overlap: say so.
                    if np.sum(good_mask) < MIN_SCALE_POINTS:
                        raise ValueError(
                            f"Cannot estimate scale between submaps {submap_id_prev} and "
                            f"{submap_id_curr}: only {int(np.sum(good_mask))} overlapping point(s) "
                            f"pass even the loosest mask (conf > 0), below the "
                            f"{MIN_SCALE_POINTS}-point floor. The submaps likely do not overlap."
                        )

            P_temp = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0]
            t1 = (P_temp[0:3,0:3] @ current_submap.get_frame_pointcloud(frame_id_curr).reshape(-1, 3)[good_mask].T).T
            t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]
            scale_factor_est_output = estimate_scale_pairwise(t1, t2)
            print(colored("scale factor", 'green'), scale_factor_est_output)
            scale_geom = scale_factor_est_output[0]
            scale_info = scale_factor_est_output[1]  # dispersion diagnostics (may be None)
            scale_factor = scale_geom
            n_overlap = int(np.sum(good_mask))

            # EXP-12 metric scale anchor (opt-in): replace the geometric overlap scale with the
            # metric-ratio cross-submap scale s_metric = r_curr / r_prev when the 4-tier sanity
            # gate trusts the anchor. Relative/ratio form only (bias-immune); default OFF.
            if self.metric_scale_anchor:
                curr_ratio = getattr(current_submap, "metric_scale", None)
                prev_ratio = getattr(prior_submap, "metric_scale", None)
                scale_factor, anchor_dec = combine_scale(
                    scale_geom, curr_ratio, prev_ratio, n_overlap,
                    gate=self.metric_anchor.gate,
                    within_cov_thresh=self.metric_anchor.within_cov_thresh,
                    thin_overlap=self.metric_anchor.thin_overlap,
                )
                self._log_anchor_decision(submap_id_curr, current_submap, anchor_dec)

            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            if DEBUG:
                print("Estimated scale factor between submaps:", scale_factor)
                debug_visualize(scale_factor*t1, t2)

            # Compute the first camera matrix of the new submap in world frame.
            H_overlap_prior_overlap_current = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0] @ H_scale
            H_w_submap = self.graph.get_homography(overlapping_node_id_prev) @ H_overlap_prior_overlap_current

            # Add first node of the new submap to the graph.
            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            # Add between factor for intra submaps constraint. With --weighted_junctions the
            # factor's covariance is scaled by measured junction quality (default OFF: the
            # base intra_submap_noise is used unchanged).
            junction_noise = self.graph.intra_submap_noise
            if self.weighted_junctions:
                dispersion = scale_info.get("dispersion") if isinstance(scale_info, dict) else None
                cov_mult = junction_covariance_multiplier(
                    dispersion=dispersion,
                    n_overlap=n_overlap,
                    image_match_ratio=image_match_ratio,
                )
                junction_noise = self.graph.scaled_noise(cov_mult)
                print(colored(f"[junction-weight] {overlapping_node_id_prev}->"
                              f"{submap_id_curr + frame_id_curr}: dispersion="
                              f"{dispersion if dispersion is not None else float('nan'):.3f} "
                              f"n_overlap={n_overlap} match_ratio={image_match_ratio} "
                              f"-> cov x{cov_mult:.2f}", "cyan"))
            self.graph.add_between_factor(overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_overlap_prior_overlap_current, junction_noise)

            if DEBUG:
                print("Adding first homography of submap: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])
                print("Adding between factor: \n", overlapping_node_id_prev, submap_id_curr + frame_id_curr, H_scale)

        else:
            assert (submap_id_curr == 0 and frame_id_curr == 0), "First added node must be submap 0 frame 0"
            self.graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)
            self.graph.add_prior_factor(submap_id_curr + frame_id_curr, H_w_submap)
            if DEBUG:
                print("Adding first homography of graph: \n", submap_id_curr + frame_id_curr, H_w_submap / H_w_submap[-1,-1])

        # Loop closure only gets intra submap constraints.
        if is_loop_closure:
            return

        # Add nodes and edges for the inner submap constraints.
        world_to_cam = current_submap.get_all_poses()
        for index, pose in enumerate(world_to_cam):
            if index == 0:
                continue

            H_inner = world_to_cam[index-1] @ np.linalg.inv(pose) # TODO Dominic, no need to take the inverse twice, just use cam_to_world
            current_node = self.graph.get_homography(submap_id_curr + index - 1) @ H_inner

            # Add node to graph.
            self.graph.add_homography(submap_id_curr + index, current_node)

            # Add between factor for inner submap constraint.
            self.graph.add_between_factor(submap_id_curr + index - 1, submap_id_curr + index, H_inner, self.graph.inner_submap_noise)

            if DEBUG:
                print("Adding homography: \n", submap_id_curr + index, current_node / current_node[-1,-1])
                print("Adding between factor: \n", submap_id_curr + index - 1, submap_id_curr + index, H_inner)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        t1 = time.time()
        images = pred_dict["images"]  # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        detected_loops = pred_dict["detected_loops"]

        depth_map = pred_dict["depth"]  # (S, H, W, 1)
        conf = pred_dict["depth_conf"]  # (S, H, W)

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        h, w = world_points.shape[1:3]
        
        # Create projection matrices
        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)


        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0
        frame_id_prev = None

        first_edge = submap_id_prev is None

        if not first_edge:
            frame_id_prev = self.map.get_latest_submap(ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()

        # Add attributes to submap and add submap to map.
        self.current_working_submap.add_all_poses(world_to_cam)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, K_4x4)
        self.current_working_submap.set_conf_masks(conf)

        # EXP-12 metric scale anchor (opt-in): precompute this submap's 1-DOF metric ratio r_i
        # now (all data — colors, per-frame camera-z pointclouds, conf masks — is present),
        # before add_edge consumes it. Fail-safe: any model error leaves metric_scale None so
        # add_edge falls back to the geometric overlap scale.
        if self.metric_anchor is not None:
            try:
                self.current_working_submap.metric_scale = self.metric_anchor.compute_submap_ratio(
                    self.current_working_submap)
                ms = self.current_working_submap.metric_scale
                print(colored(f"[metric-anchor] submap {submap_id_curr}: r_i="
                              f"{ms['r_i'] if ms['r_i'] else float('nan'):.4f} m/local "
                              f"(within-CoV {ms['within_cov']*100:.1f}%, {ms['n_frames']} kf, "
                              f"ok={ms['ok']})", "green"))
            except Exception as e:  # noqa: BLE001 — never let the anchor break SLAM
                print(colored(f"[metric-anchor] submap {submap_id_curr}: ratio failed ({e}); "
                              f"falling back to geometry", "red"))
                self.current_working_submap.metric_scale = None

        self.map.add_submap(self.current_working_submap)

        # Add all constraints for the new submap.
        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure=False)

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            cam_to_world_lc = closed_form_inverse_se3(pred_dict["extrinsic_lc"]) 
            K_4x4_lc = np.tile(np.eye(4), (2, 1, 1))
            K_4x4_lc[:, :3, :3] = pred_dict["intrinsic_lc"]
            world_to_cam_lc = np.linalg.inv(cam_to_world_lc)
            depth_map_lc = pred_dict["depth_lc"]  # (S, H, W, 1)
            conf_lc = pred_dict["depth_conf_lc"]  # (S, H, W)

            intrinsics_cam = pred_dict["intrinsic_lc"]
            

            world_points_lc = unproject_depth_map_to_point_map(depth_map_lc, pred_dict["extrinsic_lc"], intrinsics_cam)

            lc_submap_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1
            print(f"Creating new Loop closure submap with id {lc_submap_num}")
            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)
            lc_submap.add_all_frames(pred_dict["frames_lc"])
            lc_submap.set_frame_ids(pred_dict["frames_lc_names"])
            lc_submap.set_last_non_loop_frame_index(1)

            lc_submap.add_all_poses(world_to_cam_lc)
            lc_colors = (np.transpose(pred_dict["frames_lc"].cpu().numpy(), (0, 2, 3, 1)) * 255).astype(np.uint8)
            lc_submap.add_all_points(world_points_lc, lc_colors, conf_lc, self.init_conf_threshold, K_4x4_lc)
            print("Loop closure conf", conf_lc.shape)
            print(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame)
            lc_submap.set_conf_masks(conf_lc)
            self.map.add_submap(lc_submap)

            # LC junctions also carry the VGGT image_match_ratio that verified the loop —
            # a third quality signal for --weighted_junctions (None when the flag is off
            # or the ratio was not recorded; junction_quality treats None as "no signal").
            lc_match_ratio = pred_dict.get("image_match_ratio_lc")
            self.add_edge(lc_submap_num, 0, loop.query_submap_id, loop.query_submap_frame, is_loop_closure=False, image_match_ratio=lc_match_ratio)
            self.add_edge(loop.detected_submap_id, loop.detected_submap_frame, lc_submap_num, 1, is_loop_closure=True, image_match_ratio=lc_match_ratio)

    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops, clip_model=None, clip_preprocess=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Fall back to stored clip model/preprocess if not passed
        if clip_model is None:
            clip_model = self.clip_model
        if clip_preprocess is None:
            clip_preprocess = self.clip_preprocess
        t1 = time.time()
        with self.vggt_timer:
            images = load_and_preprocess_images(image_names).to(device)
        print(f"Loaded and preprocessed {len(image_names)} images in {time.time() - t1:.2f} seconds")
        print(f"Preprocessed images shape: {images.shape}")

        # print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # First submap so set new pcd num to 0
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1

        print(f"Creating new submap with id {new_pcd_num}")
        t1 = time.time()
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))
        new_submap.set_img_names(image_names)

        with self.clip_timer:
            if clip_model is not None and clip_preprocess is not None:
                image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_names)
                new_submap.set_all_semantic_vectors(image_embs)

        self.current_working_submap = new_submap
        print(f"Created new submap in {time.time() - t1:.2f} seconds")

        with torch.no_grad():
            t1 = time.time()
            with self.vggt_timer:
                predictions = model(images)
            print(f"VGGT model inference took {time.time() - t1:.2f} seconds")

        # Check for loop closures and add retrieval vectors from new submap to the database
        predictions_lc = None
        with self.loop_closure_timer:
            detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops, max_similarity_thres=self.lc_thres)
        loop_closure_frame_names = []
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
            retrieved_frames = self.map.get_frames_from_loops(detected_loops)
            with torch.no_grad():
                lc_frames = torch.stack((new_submap.get_frame_at_index(detected_loops[0].query_submap_frame), retrieved_frames[0]), axis=0)
                predictions_lc = model(lc_frames, compute_similarity=True)
                loop_closure_frame_names = [new_submap.get_img_names_at_index(detected_loops[0].query_submap_frame), 
                self.map.get_submap(detected_loops[0].detected_submap_id).get_img_names_at_index(detected_loops[0].detected_submap_frame)]

            # Visualize loop closure frames
            if DEBUG:
                imgs = lc_frames.permute(0, 2, 3, 1).cpu().numpy()  # shape -> (2, H, W, C)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                for i in range(2):
                    axes[i].imshow(imgs[i])
                    axes[i].axis('off')
                plt.tight_layout()
                plt.title("Loop Closure Frames. Left: Query Frame, Right: Retrieved Frame")
                plt.show()

        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        predictions["detected_loops"] = detected_loops
        
        if predictions_lc is not None:
            image_match_ratio = predictions_lc["image_match_ratio"]
            if image_match_ratio < 0.85:
                print(colored("Loop closure image match ratio too low, skipping loop closure", "red"))
                predictions_lc = None # We set to None to ignore the loop closure
                predictions["detected_loops"] = []
            else:
                self.graph.increment_loop_closure()
                extrinsic_lc, intrinsic_lc = pose_encoding_to_extri_intri(predictions_lc["pose_enc"], retrieved_frames[0].shape[-2:])
                predictions["extrinsic_lc"] = extrinsic_lc
                predictions["intrinsic_lc"] = intrinsic_lc
                predictions["depth_lc"] = predictions_lc["depth"]
                predictions["depth_conf_lc"] = predictions_lc["depth_conf"]
                # Record the verification ratio for the opt-in junction weighting.
                predictions["image_match_ratio_lc"] = float(image_match_ratio)

            
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor) and key != "target_tokens":
                predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
    
        if predictions_lc is not None:
            predictions["frames_lc"] = lc_frames[0:2,...]
            print(loop_closure_frame_names)
            predictions["frames_lc_names"] = loop_closure_frame_names

        return predictions