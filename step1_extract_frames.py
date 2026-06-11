import cv2
import os

def extract_key_frames(video_path, output_folder, frames_per_second=2):
    """Extract evenly spaced frames from video for VLM input"""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    interval = max(1, int(round(fps / frames_per_second))) if fps > 0 else 1
    num_frames = total_frames // interval

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
    frames = extract_key_frames("input/room_video.mp4", "output/frames", frames_per_second=2)
    print(f"\nExtracted {len(frames)} frames")