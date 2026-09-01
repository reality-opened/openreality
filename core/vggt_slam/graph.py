import gtsam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gtsam import NonlinearFactorGraph, Values, noiseModel
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4
from gtsam.symbol_shorthand import X

from vggt_slam.slam_utils import decompose_camera, normalize_to_sl4

class PoseGraph:
    def __init__(self):
        """Initialize a factor graph for Pose3 nodes with BetweenFactors."""
        self.graph = NonlinearFactorGraph()
        self.values = Values()
        inner_noise = 0.05*np.ones(15, dtype=float)
        intra_noise = 0.05*np.ones(15, dtype=float)
        self.inner_submap_noise = noiseModel.Diagonal.Sigmas(inner_noise)
        self.intra_submap_noise = noiseModel.Diagonal.Sigmas(intra_noise)
        self.intra_submap_sigmas = intra_noise  # base sigmas, for opt-in junction weighting
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 15)
        self.initialized_nodes = set()
        self.num_loop_closures = 0 # Just used for debugging and analysis

        self.auto_cal_H_mats = dict()  # Store homographies estimated by auto-calibration

        # Telemetry only — raw det(global_h) per inserted node. The normalize_to_sl4
        # calls below are disabled; before re-enabling them we need baseline evidence of
        # how often negative/near-zero dets actually occur (the SL4-singular suspects).
        self.homography_det_log = []

    def add_homography(self, key, global_h):
        """Add a new homography node to the graph."""
        self.homography_det_log.append(float(np.linalg.det(global_h)))
        # global_h = normalize_to_sl4(global_h)
        key = X(key)
        if key in self.initialized_nodes:
            print(f"SL4 {key} already exists.")
            return
        self.values.insert(key, SL4(global_h))
        self.initialized_nodes.add(key)

    def det_telemetry(self, near_zero_tol=1e-9):
        """Summary of raw homography determinants seen by add_homography. Zero-behavior
        diagnostics for the pending decision on re-enabling the disabled normalize_to_sl4
        calls in this file."""
        dets = np.asarray(self.homography_det_log, dtype=float)
        if dets.size == 0:
            return {"count": 0, "min": None, "max": None, "n_negative": 0, "n_near_zero": 0}
        return {
            "count": int(dets.size),
            "min": float(dets.min()),
            "max": float(dets.max()),
            "n_negative": int((dets < 0).sum()),
            "n_near_zero": int((np.abs(dets) < near_zero_tol).sum()),
        }

    def scaled_noise(self, cov_multiplier, base_sigmas=None):
        """Diagonal noise model with sigmas scaled by sqrt(cov_multiplier) — the opt-in
        junction-weighting hook. ``cov_multiplier`` is a covariance/variance multiplier
        (1.0 = unchanged, >1 loosens the factor). Defaults to the intra-submap junction base."""
        sig = self.intra_submap_sigmas if base_sigmas is None else np.asarray(base_sigmas, float)
        return noiseModel.Diagonal.Sigmas(sig * np.sqrt(float(cov_multiplier)))

    def add_between_factor(self, key1, key2, relative_h, noise):
        """Add a relative SL4 constraint between two nodes."""
        # relative_h = normalize_to_sl4(relative_h)
        key1 = X(key1)
        key2 = X(key2)
        if key1 not in self.initialized_nodes or key2 not in self.initialized_nodes:
            raise ValueError(f"Both poses {key1} and {key2} must exist before adding a factor.")
        self.graph.add(gtsam.BetweenFactorSL4(key1, key2, SL4(relative_h), noise))
    
    def add_prior_factor(self, key, global_h):
        # global_h = normalize_to_sl4(global_h)
        key = X(key)
        if key not in self.initialized_nodes:
            raise ValueError(f"Trying to add prior factor for key {key} but it is not in the graph.")
        self.graph.add(PriorFactorSL4(key, SL4(global_h), self.anchor_noise))

    def get_homography(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node.
        """

        auto_cal_H = np.eye(4)
        if node_id in self.auto_cal_H_mats:
            auto_cal_H  = self.auto_cal_H_mats[node_id]
        node_id = X(node_id)
        return auto_cal_H @ self.values.atSL4(node_id).matrix()

    def get_projection_matrix(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node.
        """
        homography = self.get_homography(node_id)
        projection_matrix = np.linalg.inv(homography)
        return projection_matrix

    
    def optimize(self, verbose=False):
        """Optimize the graph with Levenberg–Marquardt and print per-factor errors."""
        # Optional verbosity settings
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")
            params.setVerbosity("ERROR")

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)

        # --- Initial total error ---
        initial_error = self.graph.error(self.values)
        print(f"Initial total error: {initial_error:.6f}")

        # --- Per-factor initial error ---
        if verbose:
            print("\nInitial per-factor errors:")
            # Remember the last factor whose error actually evaluated. The summary line
            # below used to read the loop variables directly, which raises NameError on
            # an empty graph (size() == 0 -> the loop body never runs) and equally when
            # the final factor threw before binding `e`.
            last_scored = None
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(self.values)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                    last_scored = (i, factor, e)
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

            if last_scored is None:
                print("  (no factor errors to report — graph is empty)")
            else:
                last_i, last_factor, last_e = last_scored
                keys = [gtsam.DefaultKeyFormatter(k) for k in last_factor.keys()]
                print(f"Factor {last_i} connects to {keys} with error {last_e:.6f}")

        # --- Optimize ---
        result = optimizer.optimize()

        # --- Final total error ---
        final_error = self.graph.error(result)
        # print(f"\nFinal total error: {final_error:.6f}")

        # --- Per-factor final error ---
        if verbose:
            print("\nFinal per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(result)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

        # --- Store optimized values ---
        self.values = result


    def print_estimates(self):
        """Print the optimized poses."""
        for key in sorted(self.initialized_nodes):
            print(f"Homography{key}:\n{self.values.atSL4(key)}\n")
    
    def increment_loop_closure(self):
        """Increment the loop closure count."""
        self.num_loop_closures += 1
    
    def get_num_loops(self):
        """Get the number of loop closures."""
        return self.num_loop_closures

    def update_all_homographies(self, map, auto_cal_H_mats):
        count = 0
        for submap in map.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                self.auto_cal_H_mats[id] = np.linalg.inv(auto_cal_H_mats[count])
                count += 1
        assert count == len(auto_cal_H_mats), "Number of auto-calibration homographies does not match number of poses in the map."