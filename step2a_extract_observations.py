import ollama
import os
import base64

def _encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _analyze_frame(image_path, frame_num, total):
    print(f"    Frame {frame_num}/{total}: {os.path.basename(image_path)}")
    img = _encode(image_path)
    
    combined_prompt = """Look at this indoor photo and answer in one short sentence (max 40 words).

Tell me: what type of room is this (kitchen, living room, bedroom, hallway, bathroom, staircase)? How many doors are visible? Is a staircase visible? Can you see into another room through a doorway, and if so what type?

Example answer: "Living room, medium, 2 doors on left and right walls, staircase visible, doorway leads to kitchen."
"""
    try:
        resp = ollama.chat(
            model="moondream",
            messages=[{"role": "user", "content": combined_prompt, "images": [img]}]
        )
        answer = resp["message"]["content"].strip()
    except Exception as e:
        answer = f"Error processing frame: {e}"

    return {"frame": frame_num, "details": answer}

def extract_observations(frames_folder):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    total = len(frame_files)

    print(f"  Phase A: Analyzing {total} frames with Moondream (image model)...")

    observations = []
    summary = ""
    for i, fname in enumerate(frame_files):
        obs = _analyze_frame(os.path.join(frames_folder, fname), i + 1, total)
        observations.append(obs)
        
        frame_text = f"\nFrame {obs['frame']}:\n  Details:\n{obs['details']}\n"
        print(f"    Summary:\n{obs['details']}\n")
        
        if len(summary) + len(frame_text) > 12000:
            summary += "\n[Warning: Further frames truncated to prevent memory overflow]\n"
            break
        summary += frame_text

    os.makedirs("output", exist_ok=True)
    with open("output/observations.txt", "w", encoding="utf-8") as f:
        f.write(summary)
        
    print("\n[OK] Observations saved to output/observations.txt")

if __name__ == "__main__":
    frames_dir = "output/frames"
    if not os.path.exists(frames_dir):
        print(f"Error: {frames_dir} not found. Please run step1_extract_frames.py first.")
        import sys
        sys.exit(1)
        
    extract_observations(frames_dir)
