"""
Indoor Navigation -- Video to Floor Plan Pipeline
==================================================
Master script that runs the entire pipeline end-to-end:

    Video (mp4) -> Depth Estimation -> 3D Point Cloud -> 2D Occupancy -> AI Cleaning -> Floor Plan

Usage:
    python run_pipeline.py --video input.mp4
    python run_pipeline.py --video walkthrough.mp4 --fps 3 --max-frames 20

This is the ONLY script you need to run. It will:
    1. Run 3D reconstruction (Depth Anything V2 + Open3D ICP)
    2. Project to 2D occupancy map
    3. Clean the map using the trained U-Net model
    4. Output a clean, architectural floor plan image
"""

import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(
        description='Convert a walking video into a clean 2D floor plan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --video input.mp4
  python run_pipeline.py --video walkthrough.mp4 --fps 3
  python run_pipeline.py --video corridor.mp4 --max-frames 15
        """
    )
    parser.add_argument('--video', type=str, default='input.mp4',
                        help='Path to the input walkthrough video (default: input.mp4)')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second to process (lower = faster, default: 2)')
    parser.add_argument('--max-frames', type=int, default=30,
                        help='Maximum frames to process (default: 30)')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Path to trained U-Net model (default: best_model.pth)')
    parser.add_argument('--skip-3d', action='store_true',
                        help='Skip 3D reconstruction (use existing raw_occupancy_map.png)')
    args = parser.parse_args()

    total_start = time.time()

    print()
    print("=" * 60)
    print("  Indoor Navigation -- Video to Floor Plan Pipeline")
    print("  Powered by Depth Anything V2 + U-Net Deep Learning")
    print("=" * 60)
    print()

    # =====================
    # Check prerequisites
    # =====================
    if not args.skip_3d and not os.path.exists(args.video):
        print(f"ERROR: Video file '{args.video}' not found!")
        print(f"Please place your walkthrough video in this folder and try again.")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"WARNING: U-Net model '{args.model}' not found!")
        print(f"The floor plan cleaning step will be skipped.")
        print(f"To train the model, run: python train.py")
        print()

    # =====================
    # Step 1: 3D Reconstruction
    # =====================
    if not args.skip_3d:
        print("-" * 60)
        print("  STEP 1: 3D Reconstruction from Video")
        print("-" * 60)
        
        from reconstruct import reconstruct_from_video, project_to_2d
        
        global_pcd = reconstruct_from_video(
            args.video,
            target_fps=args.fps,
            max_frames=args.max_frames
        )
        
        if global_pcd is None:
            print("FATAL: 3D reconstruction failed. Exiting.")
            sys.exit(1)
        
        occupancy = project_to_2d(global_pcd, output_path='raw_occupancy_map.png')
        
        if occupancy is None:
            print("FATAL: 2D projection failed. Exiting.")
            sys.exit(1)
    else:
        print("  Skipping 3D reconstruction (using existing raw_occupancy_map.png)")
        if not os.path.exists('raw_occupancy_map.png'):
            print("ERROR: raw_occupancy_map.png not found! Cannot skip 3D step.")
            sys.exit(1)

    # =====================
    # Step 2: AI Floor Plan Cleaning
    # =====================
    if os.path.exists(args.model):
        print("-" * 60)
        print("  STEP 2: AI Floor Plan Cleaning (U-Net)")
        print("-" * 60)
        
        from clean_floorplan import clean_floor_plan
        
        floor_plan = clean_floor_plan(
            occupancy_input='raw_occupancy_map.png',
            model_path=args.model
        )
        
        if floor_plan is None:
            print("WARNING: AI cleaning failed. Raw occupancy map is still available.")
    else:
        print()
        print("  Skipping AI cleaning (no trained model found).")
        print("  The raw occupancy map has been saved as raw_occupancy_map.png")

    # =====================
    # Summary
    # =====================
    total_time = time.time() - total_start
    
    print()
    print("=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Total Time: {total_time:.1f} seconds")
    print()
    print("  Output Files:")
    print("    - full_reconstruction.ply  -- 3D point cloud")
    print("    - raw_occupancy_map.png    -- Raw 2D projection")
    
    if os.path.exists(args.model):
        print("    - floor_plan.png          -- Clean 2D floor plan")
        print("    - pipeline_result.png     -- Full pipeline visual")
    
    print()
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
