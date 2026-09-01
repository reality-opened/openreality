import os
import glob
import time
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap

from vggt.models.vggt import VGGT

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--export_dataset", type=str, default=None, help="If set, write an OpenReality robot-training export (trajectory + video + cloud + grounding sidecar) under this directory after SLAM finishes. Requires the sibling server repo on the path (reality-opened/server); see its docs/dataset-export.md")
parser.add_argument("--export_fps", type=float, default=10.0, help="Synthetic uniform fps for the exported trajectory timeline (keyframes are motion-gated). See docs/dataset-export.md §7.3 in the server repo (reality-opened/server)")
parser.add_argument("--export_isaac", type=str, default=None, help="If set, write an Isaac Sim/Lab USD scene (metric, gravity-aligned, Z-up: scene.usd + trajectory.usd + manifest.json) under this directory after SLAM finishes. Requires the sibling server repo on the path (reality-opened/server); see its docs/isaac-export.md")
parser.add_argument("--isaac_scale", type=float, default=None, help="SLAM-units -> meters scale factor for the Isaac export. Without it (or --isaac_ref_object/--isaac_ref_meters) the scene stays up-to-scale (flagged metric=false).")
parser.add_argument("--isaac_ref_object", type=str, default=None, help="Reference object query (e.g. 'door') whose known real size sets the Isaac metric scale; pair with --isaac_ref_meters. Requires --run_os so the scene graph exists.")
parser.add_argument("--isaac_ref_meters", type=float, default=None, help="Real-world size (meters, along the gravity/up axis) of --isaac_ref_object, used to back out the metric scale.")
parser.add_argument("--isaac_no_mesh", action="store_true", help="Skip mesh reconstruction for the Isaac export and write a points-only scene (no collision).")
parser.add_argument("--export_splat", type=str, default=None, help="If set, write a standard 3DGS splat.ply (point cloud -> gaussians) at this path after SLAM finishes. Deterministic, CPU-only, no training. See docs/splat-export.md")
parser.add_argument("--export_splat_refine", action="store_true", help="Optional GPU photometric refinement of --export_splat with gsplat (guarded; falls back to the baseline init if gsplat/CUDA/keyframes are unavailable). See docs/splat-export.md")
parser.add_argument("--no_splat_scale_clamp", action="store_true", help="Disable the export-time p99 per-gaussian scale-tail clamp on --export_splat (default ON; the clamp removes the refined-splat render spikes and is a near-no-op on the baseline). See docs/splat-export.md")
parser.add_argument("--splat_scale_clamp_pct", type=float, default=None, help="Percentile for the --export_splat scale-tail clamp (default 99). Ignored with --no_splat_scale_clamp.")
parser.add_argument("--metric_scale_anchor", action="store_true", help="EXP-12: per-submap metric-ratio scale anchor (Depth-Anything-V2-Metric-Hypersim-Large, Apache-2.0) that collapses cross-submap scale drift. Default OFF. RELATIVE drift removal only — the model carries a measured +16.5%% absolute bias, so absolute metres are NOT claimed. See experiments/research/2026-07-06-exp12-metric-anchor.md")
parser.add_argument("--metric_anchor_keyframes", type=int, default=3, help="K representative keyframes per submap for the metric anchor (default 3). Ignored without --metric_scale_anchor.")
parser.add_argument("--metric_anchor_gate", type=float, default=0.5, help="Metric-anchor agreement gate: max |ln(s_metric/scale_geom)| to auto-trust the anchor (default 0.5). Ignored without --metric_scale_anchor.")
parser.add_argument("--metric_anchor_min_valid", type=int, default=200, help="Minimum shared valid pixels per keyframe for a metric-anchor ratio (default 200). Ignored without --metric_scale_anchor.")
parser.add_argument("--weighted_junctions", action="store_true", help="Scale each cross-submap junction factor's covariance by measured junction quality (scale-estimate dispersion, overlap conf-mask coverage, LC image_match_ratio) so weak junctions get looser factors the graph can override. Default OFF. Mapping lives in vggt_slam/junction_weight.py")


def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        metric_scale_anchor=args.metric_scale_anchor,
        metric_anchor_keyframes=args.metric_anchor_keyframes,
        metric_anchor_gate=args.metric_anchor_gate,
        metric_anchor_min_valid=args.metric_anchor_min_valid,
        weighted_junctions=args.weighted_junctions,
    )

    print("Initializing and loading VGGT model...")


    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, clip_model, clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    # Export the scan as an OpenReality robot-training dataset (server repo,
    # docs/dataset-export.md).
    # Runs regardless of the interactive --run_os query loop below, which never returns.
    if args.export_dataset:
        # The exporters live in the sibling `server` repo, not here — importing them is an
        # inverted dependency that only resolves when core is checked out inside the
        # platform super-repo. Fail with that explanation rather than a bare ModuleNotFound.
        try:
            from server.export.writer import export_scene
        except ImportError as exc:
            raise ImportError(
                "--export_dataset requires running inside the platform super-repo with the "
                "server repo importable; the exporters live in reality-opened/server "
                "(server/export/writer.py), not in this repo (core). See that repo's "
                "docs/dataset-export.md."
            ) from exc
        export_dir = export_scene(
            solver,
            detections=[],
            report=None,
            out_dir=args.export_dataset,
            fps=args.export_fps,
        )
        print(f"Exported OpenReality dataset to {export_dir}")

    # Export the scan as an Isaac Sim/Lab USD scene (server repo, docs/isaac-export.md).
    if args.export_isaac:
        # Same inverted dependency as --export_dataset above.
        try:
            from server.export.isaac.writer import export_isaac
        except ImportError as exc:
            raise ImportError(
                "--export_isaac requires running inside the platform super-repo with the "
                "server repo importable; the exporters live in reality-opened/server "
                "(server/export/isaac/writer.py), not in this repo (core). See that repo's "
                "docs/isaac-export.md."
            ) from exc
        isaac_dir = export_isaac(
            solver,
            out_dir=args.export_isaac,
            fps=args.export_fps,
            scale=args.isaac_scale,
            ref_object=args.isaac_ref_object,
            ref_meters=args.isaac_ref_meters,
            reconstruct=not args.isaac_no_mesh,
        )
        if isaac_dir:
            print(f"Exported Isaac USD scene to {isaac_dir}")

    # Export the SLAM result as a standard 3DGS splat.ply (docs/splat-export.md).
    # Baseline is deterministic/CPU-only; --export_splat_refine adds optional guarded
    # gsplat GPU refinement that silently falls back to the baseline if unavailable.
    if args.export_splat:
        from vggt_slam.splat_export import export_splat, DEFAULT_SCALE_CLAMP_PERCENTILE
        splat_path = export_splat(
            solver, args.export_splat, refine=args.export_splat_refine,
            overlap_frames=args.overlapping_window_size,
            clamp_scales=not args.no_splat_scale_clamp,
            scale_clamp_percentile=(args.splat_scale_clamp_pct
                                    if args.splat_scale_clamp_pct is not None
                                    else DEFAULT_SCALE_CLAMP_PERCENTILE),
        )
        if splat_path:
            print(f"Exported 3DGS splat to {splat_path}")

    if args.run_os:
        while True:
            # Prompt user for text input
            query = input("\nEnter text query or q to quit: ").strip()
            if len(query) == 0:
                print("Empty query. Exiting.")
                return
            
            if query == "q":
                print("Exiting.")
                return
            
            start_time = time.time()
            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            overall_best_score, overall_best_submap_id, overall_best_frame_index = solver.map.retrieve_best_semantic_frame(text_emb)

            found_submap = solver.map.get_submap(overall_best_submap_id)

            # Display image
            best_img = found_submap.get_frame_at_index(overall_best_frame_index)
            print("Score:", overall_best_score)
            with torch.no_grad():
                # convert torch image to PIL
                best_img = to_pil_image(best_img)
                inference_state = processor.set_image(best_img)
                output = processor.set_text_prompt(state=inference_state, prompt=query)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                print("Scores:", scores.cpu().numpy())

            print("Time taken for query:", time.time() - start_time)

            masked_img = utils.overlay_masks(best_img, masks)
            masked_img.show()

            for i in range(masks.shape[0]):
                mask = masks[i].cpu().numpy()
                obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                solver.viewer.visualize_obb(
                    center=obb_center,
                    extent=obb_extent,
                    rotation=obb_rotation,
                    color=(255, 0, 0),
                    line_width=8.0,
                )

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        # Log the full point cloud as one file, used for visualization.
        # solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            solver.map.save_framewise_pointclouds(solver.graph, args.log_path.replace(".txt", "_logs"))


if __name__ == "__main__":
    main()
