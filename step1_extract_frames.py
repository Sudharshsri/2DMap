import cv2
import os

def extract_key_frames(video_path, output_folder, frames_per_second=2):
    """Extract evenly spaced frames from video for VLM input"""
    os.makedirs(output_folder, exist_ok=True)
    
    # CRITICAL FIX: Clear old frames from the folder so previous runs don't mix in!
    for f in os.listdir(output_folder):
        if f.endswith(".jpg"):
            os.remove(os.path.join(output_folder, f))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    interval = max(1, int(round(fps / frames_per_second))) if fps > 0 else 1
    
    # Use ceiling division to ensure we capture the very last partial second
    import math
    num_frames = math.ceil(total_frames / interval) if interval > 0 else 0

    saved = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(output_folder, f"frame_{i:04d}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)
            print(f"  Saved: {path}")

    cap.release()
    return saved

if __name__ == "__main__":
    # Extract frames independently
    frames = extract_key_frames("input/room_video.mp4", "output/frames", frames_per_second=1)
    
    print(f"\n✅ Extracted {len(frames)} frames to output/frames/")