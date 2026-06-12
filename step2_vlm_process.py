import ollama
import json
import os
import base64

def _encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Phase A: Visual analysis — Moondream (good at image Q&A) ─────────────────

def _analyze_frame(image_path, frame_num, total):
    """Ask Moondream ONE combined question to save processing time"""
    print(f"    Frame {frame_num}/{total}: {os.path.basename(image_path)}")
    img = _encode(image_path)

    # COMBINED PROMPT: 1 API call instead of 3
    combined_prompt = """You are analyzing a single frame from a continuous indoor home walkthrough video.

Your task is to describe only the architectural and spatial information useful for reconstructing a floor plan.

Write EXACTLY TWO SHORT PARAGRAPHS.

Paragraph 1:
Describe the currently visible space.
Include:
- probable room type (living room, bedroom, kitchen, bathroom, corridor, staircase, balcony, etc.)
- approximate room size (small, medium, large)
- visible walls
- visible doors
- visible windows
- major openings or passages
- any architectural features relevant to room layout

Paragraph 2:
Describe the spatial relationships visible in this frame.
Include:
- where doors/openings appear to lead
- corridors or connecting spaces
- staircases and their direction
- apparent camera movement direction (forward, backward, left, right)
- whether the camera appears to be entering, leaving, or remaining in the current room

Rules:
- Focus only on structure and layout.
- Ignore furniture unless it helps identify the room type.
- Do not estimate dimensions.
- Do not invent rooms that are not visible.
- If uncertain, explicitly say "unclear".
- Keep total response under 80 words.
- Do not use bullet points.
- Do not use headings.
"""

    try:
        resp = ollama.chat(
            model="moondream",
            messages=[{"role": "user", "content": combined_prompt, "images": [img]}]
        )
        answer = resp["message"]["content"].strip()
    except Exception as e:
        answer = f"Error processing frame: {e}"

    # Return the consolidated 'details' 
    return {"frame": frame_num, "details": answer}


# ── Phase B: JSON synthesis — llama3.2:3b (good at structured text output) ───

def _synthesize(observations, is962_context):
    
    # SAFER TRUNCATION: Find the last period (.) before the 600th character
    if len(is962_context) > 600:
        cut_point = is962_context.rfind('.', 0, 600)
        is962_short = is962_context[:cut_point + 1] if cut_point != -1 else is962_context[:600]
    else:
        is962_short = is962_context

    # Build summary, capping it to prevent context overflow (Llama crash)
    summary = ""
    for obs in observations:
        frame_text = f"\nFrame {obs['frame']}:\n  Details: {obs['details']}\n"
        if len(summary) + len(frame_text) > 4000: 
            summary += "\n[Warning: Further frames truncated to prevent memory overflow]\n"
            break
        summary += frame_text
    schema = """
{
  "metadata": {
    "title": "HOME FLOOR PLAN",
    "units": "mm",
    "standard": "IS962",
    "confidence": 0.0
  },
  "rooms": [
    {
      "id": "room1",
      "name": "Living Room",
      "x": 0,
      "y": 0,
      "width": 100,
      "height": 100
    }
  ],
  "doors": [
    {
      "room_id": "room1",
      "wall": "right",
      "position_ratio": 0.5,
      "width": 15,
      "swing": "left"
    }
  ],
  "windows": [
    {
      "room_id": "room1",
      "wall": "top",
      "position_ratio": 0.3,
      "width": 20
    }
  ],
  "corridors": [],
  "stairs": [],
  "adjacency_graph": {}
}
"""
    prompt = f"""
Using the frame observations, reconstruct the logical layout of the house.
1. Invent approximate, logical 2D numerical coordinates (x, y, width, height) for every room to create a basic top-down floor plan layout.
2. START AT THE BOTTOM: The first room seen in the video (the entrance) must be placed at the bottom of the map (e.g., y=200). 
3. As the camera moves forward, build new rooms upwards (decrease the Y coordinate). If the camera moves left/right, build rooms to the sides (decrease/increase the X coordinate).
4. Ensure connected rooms are placed adjacent to each other in the coordinate space.
5. Place doors on the appropriate walls (top, bottom, left, right) with a position_ratio between 0.0 and 1.0.

FRAME OBSERVATIONS:

{summary}

IS962 CONTEXT:

{is962_short}

Output schema:

{schema}

Output ONLY valid JSON matching the exact schema structure.
"""

    for attempt in range(1, 4):
        try:
            resp = ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            raw = resp["message"]["content"].strip()

            # Your original robust markdown parsing logic
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    p = part.strip()
                    if p.startswith("json"):
                        raw = p[4:].strip()
                        break
                    elif p.startswith("{"):
                        raw = p
                        break

            # Find standard JSON boundaries
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            data = json.loads(raw)
            print(f"  JSON ready (attempt {attempt})")
            return data

        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < 3:
                print("  Retrying...")

    print("  WARNING: Could not produce valid JSON. Using fallback template.")
    return _fallback()


def _fallback():
    return {
        "metadata": {"title": "FLOOR PLAN", "scale": "NTS", "sheet": "A4", "orientation": "landscape"},
        "rooms":   [{"id": "R1", "name": "Room", "x": 40, "y": 60, "width": 200, "height": 120}],
        "doors":   [{"room_id": "R1", "wall": "bottom", "position_ratio": 0.5, "width": 12, "swing": "left"}],
        "windows": [{"room_id": "R1", "wall": "right", "position_ratio": 0.3, "width": 15}]
    }


# ── Public entry point ────────────────────────────────────────────────────────

def process_frames_with_vlm(frames_folder, is962_context):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    total = len(frame_files)

    print(f"  Phase A: Analyzing {total} frames with Moondream (image model)...")

    observations = []
    for i, fname in enumerate(frame_files):
        # Pass standard os paths 
        obs = _analyze_frame(os.path.join(frames_folder, fname), i + 1, total)
        observations.append(obs)
        print(f"    Summary:\n{obs['details']}\n")

    print("\n  Phase B: Synthesizing JSON with llama3.2:3b (text model)...")
    return _synthesize(observations, is962_context)


if __name__ == "__main__":
    import sys
    
    context_path = "output/is962_context.txt"
    if not os.path.exists(context_path):
        print(f"Error: {context_path} not found. Please run step0_parse_is_962.py first.")
        sys.exit(1)
        
    with open(context_path, "r", encoding="utf-8") as f:
        ctx = f.read()
        
    data = process_frames_with_vlm("output/frames", ctx)
    
    # Save the floor plan JSON
    os.makedirs("output", exist_ok=True)
    
    with open("output/floor_plan.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print("\n✅ Floor plan JSON saved to output/floor_plan.json")