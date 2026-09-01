"""
Quick Viser viewer for modal_results/poses_logs/*.npz point clouds.
Usage: python visualize_results.py [--logs_dir modal_results/poses_logs] [--port 8080]
"""
import argparse
import glob
import time

import numpy as np
import viser

def height_colormap(pts: np.ndarray) -> np.ndarray:
    """Color points by Z height (blue=low, red=high)."""
    z = pts[:, 2]
    z_min, z_max = np.percentile(z, 2), np.percentile(z, 98)
    if z_max == z_min:
        return np.full((len(pts), 3), 128, dtype=np.uint8)
    t = np.clip((z - z_min) / (z_max - z_min), 0, 1)
    # Blue → Cyan → Green → Yellow → Red
    colors = np.zeros((len(pts), 3), dtype=np.uint8)
    colors[:, 0] = (np.clip(t * 2 - 1, 0, 1) * 255).astype(np.uint8)   # R
    colors[:, 1] = (np.clip(1 - np.abs(t * 2 - 1), 0, 1) * 255).astype(np.uint8)  # G
    colors[:, 2] = (np.clip(1 - t * 2, 0, 1) * 255).astype(np.uint8)   # B
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="modal_results/poses_logs")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--voxel_size", type=float, default=0.02,
                        help="Downsample voxel size (0 = no downsampling)")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.logs_dir}/*.npz"))
    if not files:
        print(f"No .npz files found in {args.logs_dir}")
        return
    print(f"Loading {len(files)} point cloud frames...")

    all_pts = []
    for fp in files:
        d = np.load(fp)
        pts = d["pointcloud"]   # (H, W, 3)
        mask = d["mask"]        # (H, W)
        pts_flat = pts[mask]    # (N, 3)
        if len(pts_flat):
            all_pts.append(pts_flat.astype(np.float32))

    if not all_pts:
        print("No valid points after masking.")
        return

    pts = np.concatenate(all_pts, axis=0)
    print(f"Total points: {len(pts):,}")

    # Optional voxel downsampling
    if args.voxel_size > 0:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(args.voxel_size)
        pts = np.asarray(pcd.points, dtype=np.float32)
        print(f"After voxel downsampling ({args.voxel_size}m): {len(pts):,} points")

    colors = height_colormap(pts)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    print(f"\nViser server running — open http://localhost:{args.port} in your browser\n")

    server.scene.add_point_cloud(
        name="slam_map",
        points=pts,
        colors=colors,
        point_size=0.01,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down.")


if __name__ == "__main__":
    main()
