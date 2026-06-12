import cv2
import os
from pathlib import Path

def extract_key_frames(video_path, output_folder, frames_per_second):
    """Extract evenly spaced frames from video for VLM input"""
    
    # Use pathlib for cleaner, object-oriented path handling
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old frames robustly
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    # Use a try...finally block to guarantee the capture object is released 
    # even if the script encounters an error during processing.
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate how many frames to skip. Max ensures we don't divide by zero.
        interval = max(1, int(round(fps / frames_per_second))) if fps > 0 else 1

        saved = []
        frame_counter = 0
        extracted_counter = 0

        # Sequential reading is significantly faster and more accurate than cap.set()
        while True:
            # cap.grab() points to the next frame but does not decode it, saving CPU time
            ret = cap.grab()
            if not ret:
                break # End of video
            
            # If the current frame falls on our desired interval, decode and save it
            if frame_counter % interval == 0:
                ret, frame = cap.retrieve()
                if ret:
                    # Save using zero-padded format
                    path = out_dir / f"frame_{extracted_counter:04d}.jpg"
                    cv2.imwrite(str(path), frame)
                    saved.append(str(path))
                    print(f"  Saved: {path}")
                    extracted_counter += 1
            
            frame_counter += 1

    finally:
        # Resource cleanup is guaranteed
        cap.release()

    return saved

if __name__ == "__main__":
    import sys
    import glob

    # Accept video path as CLI arg, otherwise auto-detect latest .mp4 in input/
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
    else:
        candidates = sorted(glob.glob("input/*.mp4"), key=os.path.getmtime, reverse=True)
        if not candidates:
            print("Error: No .mp4 files found in input/")
            sys.exit(1)
        video_source = candidates[0]

    print(f"  Using video: {video_source}")
    output_dest = "output/frames"
    frames_per_second = 0.5
    frames = extract_key_frames(video_source, output_dest, frames_per_second)

    print(f"\n[OK] Extracted {len(frames)} frames to {output_dest}/")