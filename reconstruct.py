"""
Video to 3D Reconstruction Pipeline
=====================================
Takes an input video (mp4) of someone walking through an indoor space
and produces a 3D point cloud reconstruction of the environment.

Pipeline:
    1. Extract frames from video at a configurable FPS
    2. Run Depth Anything V2 (Small) to estimate depth per frame
    3. Back-project RGB + Depth into 3D point clouds (Open3D)
    4. Register (stitch) consecutive point clouds using ICP
    5. Save the merged global point cloud as .ply

Usage:
    python reconstruct.py --video input.mp4 --fps 2
"""

import cv2
import torch
import numpy as np
import open3d as o3d
import os
import argparse
import time

from transformers import pipeline as hf_pipeline
from PIL import Image


def load_depth_model():
    """Load the Depth Anything V2 Small model via HuggingFace."""
    print("  Loading Depth Anything V2 (Small) model...")
    print("  (First run will download ~100MB of model weights)")
    print()
    depth_estimator = hf_pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=-1  # Force CPU
    )
    return depth_estimator


def estimate_depth(depth_estimator, frame_bgr, target_width, target_height):
    """
    Run depth estimation on a single BGR frame.
    Returns np.ndarray of shape (H, W) with float32 depth values.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    
    result = depth_estimator(pil_img)
    
    depth_pil = result["depth"].resize((target_width, target_height))
    depth_map = np.array(depth_pil).astype(np.float32)
    
    # Depth Anything outputs disparity-like values (higher = closer)
    # Invert so higher = further for Open3D
    depth_max = depth_map.max()
    if depth_max > 0:
        depth_map = depth_max - depth_map
    
    # Clamp to avoid zero depths
    depth_map = np.clip(depth_map, 1.0, None)
    
    return depth_map


def create_point_cloud(rgb_bgr, depth_map, intrinsic):
    """Create an Open3D point cloud from an RGB frame and depth map."""
    color = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    
    # Normalize depth to a reasonable metric scale
    d_min = depth_map.min()
    d_max = depth_map.max()
    if d_max > d_min:
        depth_normalized = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_normalized = np.zeros_like(depth_map)
    
    # Scale to room-sized depth range (0.5m to 6m)
    depth_metric = (depth_normalized * 5.5 + 0.5).astype(np.float32)
    
    o3d_color = o3d.geometry.Image(color)
    o3d_depth = o3d.geometry.Image(depth_metric)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=1.0,
        depth_trunc=8.0,
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    # Flip Y and Z to correct orientation
    pcd.transform([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    
    return pcd


def preprocess_pcd(pcd, voxel_size):
    """Downsample and compute features for registration."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh


def register_point_clouds(source, target, voxel_size):
    """
    Register (align) source to target using:
        1. Fast Global Registration (coarse alignment)
        2. Point-to-Plane ICP (fine refinement)
    Returns 4x4 transformation matrix.
    """
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    
    source_down, source_fpfh = preprocess_pcd(source, voxel_size)
    target_down, target_fpfh = preprocess_pcd(target, voxel_size)
    
    # Step 1: Fast Global Registration
    dist_threshold = voxel_size * 1.5
    result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=dist_threshold
        )
    )
    
    # Step 2: ICP refinement
    dist_threshold_icp = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, dist_threshold_icp,
        result_fgr.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    return result_icp.transformation


def reconstruct_from_video(video_path, target_fps=2, max_frames=30):
    """
    Main reconstruction pipeline.
    
    Args:
        video_path: Path to input .mp4 video
        target_fps: How many frames per second to process
        max_frames: Maximum number of frames to process
    
    Returns:
        global_pcd: The merged Open3D PointCloud
    """
    print()
    print("=" * 60)
    print("  3D Reconstruction Pipeline")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print(f"  Target FPS: {target_fps}")
    print(f"  Max Frames: {max_frames}")
    print("=" * 60)
    print()
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file '{video_path}' not found!")
        return None
    
    # Load depth model
    depth_estimator = load_depth_model()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Downscale for CPU performance
    scale = 1.0
    if width > 640:
        scale = 640 / width
    proc_width = int(width * scale)
    proc_height = int(height * scale)
    
    print(f"  Video FPS: {video_fps:.1f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Original Resolution: {width}x{height}")
    print(f"  Processing Resolution: {proc_width}x{proc_height}")
    
    # Frame skip interval
    frame_skip = max(1, int(video_fps / target_fps))
    print(f"  Processing every {frame_skip}th frame")
    print()
    
    # Camera intrinsics (reasonable estimate for a smartphone)
    focal_length = proc_width * 0.8
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        proc_width, proc_height,
        focal_length, focal_length,
        proc_width / 2, proc_height / 2
    )
    
    # =====================
    # Phase 1: Extract Point Clouds
    # =====================
    print("Phase 1: Extracting 3D point clouds from video frames...")
    point_clouds = []
    frame_idx = 0
    processed = 0
    
    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            t0 = time.time()
            
            # Resize
            if scale < 1.0:
                frame = cv2.resize(frame, (proc_width, proc_height))
            
            # Depth estimation
            depth_map = estimate_depth(depth_estimator, frame, proc_width, proc_height)
            
            # Create point cloud
            pcd = create_point_cloud(frame, depth_map, intrinsic)
            
            # Clean up: downsample + remove outliers
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            if len(pcd.points) > 100:
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                pcd = pcd.select_by_index(ind)
            
            if len(pcd.points) > 50:
                point_clouds.append(pcd)
                processed += 1
                dt = time.time() - t0
                print(f"  Frame {frame_idx:4d} -> {len(pcd.points):5d} points ({dt:.1f}s)")
        
        frame_idx += 1
    
    cap.release()
    
    if len(point_clouds) == 0:
        print()
        print("ERROR: No valid point clouds extracted!")
        return None
    
    print(f"\n  Extracted {len(point_clouds)} point clouds.\n")
    
    # =====================
    # Phase 2: Stitch Point Clouds
    # =====================
    print("Phase 2: Stitching point clouds (ICP Registration)...")
    global_pcd = point_clouds[0]
    voxel_size = 0.1
    
    for i in range(1, len(point_clouds)):
        t0 = time.time()
        source = point_clouds[i]
        
        try:
            transformation = register_point_clouds(source, global_pcd, voxel_size)
            source.transform(transformation)
            global_pcd += source
            
            # Keep the merged cloud manageable
            global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)
            
            dt = time.time() - t0
            print(f"  Stitched {i}/{len(point_clouds)-1} ({len(global_pcd.points)} total points, {dt:.1f}s)")
        except Exception as e:
            print(f"  WARNING: Failed to stitch frame {i}: {e}")
            continue
    
    # Final cleanup
    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.03)
    if len(global_pcd.points) > 100:
        cl, ind = global_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        global_pcd = global_pcd.select_by_index(ind)
    
    # Save
    output_ply = 'full_reconstruction.ply'
    o3d.io.write_point_cloud(output_ply, global_pcd)
    print(f"\n  [OK] Saved 3D point cloud: {output_ply}")
    print(f"  Total points: {len(global_pcd.points)}")
    print()
    
    return global_pcd


def project_to_2d(global_pcd, output_path='raw_occupancy_map.png', grid_resolution=400):
    """
    Project the 3D point cloud top-down to create a 2D occupancy grid.
    """
    print("Phase 3: Projecting 3D -> 2D occupancy map...")
    
    points = np.asarray(global_pcd.points)
    
    if len(points) == 0:
        print("  ERROR: No points to project!")
        return None
    
    # Use Y-axis for height filtering (keep middle 60% of points)
    y_vals = points[:, 1]
    y_lo = np.percentile(y_vals, 20)
    y_hi = np.percentile(y_vals, 80)
    height_mask = (y_vals > y_lo) & (y_vals < y_hi)
    sliced = points[height_mask]
    
    if len(sliced) < 10:
        print("  WARNING: Very few points after height slicing. Using all points.")
        sliced = points
    
    # Project onto X-Z plane
    x = sliced[:, 0]
    z = sliced[:, 2]
    
    # Create 2D histogram
    H, xedges, zedges = np.histogram2d(x, z, bins=grid_resolution)
    
    # Normalize and binarize
    occupancy = (H > 0).astype(np.float32)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    occupancy_uint8 = (occupancy * 255).astype(np.uint8)
    
    # Close small gaps
    occupancy_uint8 = cv2.morphologyEx(occupancy_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Remove tiny noise blobs
    occupancy_uint8 = cv2.morphologyEx(occupancy_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    
    occupancy = (occupancy_uint8 > 0).astype(np.float32)
    
    # Save
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(occupancy.T, origin='lower', cmap='gray_r')
    ax.set_title('Raw 2D Occupancy Map (from 3D Reconstruction)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved raw occupancy map: {output_path}")
    print(f"  Grid size: {grid_resolution}x{grid_resolution}")
    print()
    
    return occupancy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct 3D from video')
    parser.add_argument('--video', type=str, default='input.mp4', help='Path to input video')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second to process')
    parser.add_argument('--max-frames', type=int, default=30, help='Max frames to process')
    args = parser.parse_args()
    
    pcd = reconstruct_from_video(args.video, target_fps=args.fps, max_frames=args.max_frames)
    
    if pcd is not None:
        project_to_2d(pcd)
