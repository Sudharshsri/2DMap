import ollama
import json
import os
import base64

def _encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Phase A: Visual analysis — Moondream (good at image Q&A) ─────────────────

def _analyze_frame(image_path, frame_num, total):
    """Ask Moondream 3 focused questions about a single frame"""
    print(f"    Frame {frame_num}/{total}: {os.path.basename(image_path)}")
    img = _encode(image_path)

    questions = [
        "What rooms or spaces are visible in this image? "
        "List each room name and describe its size as small/medium/large "
        "relative to the others.",

        "Are there any doors visible? For each door say: "
        "room name, which wall (left/right/top/bottom), "
        "position along wall (start/middle/end).",

        "Are there any windows visible? For each window say: "
        "room name, which wall, position along wall (start/middle/end)."
    ]

    answers = []
    for q in questions:
        try:
            resp = ollama.chat(
                model="moondream",
                messages=[{"role": "user", "content": q, "images": [img]}]
            )
            answers.append(resp["message"]["content"].strip())
        except Exception as e:
            answers.append(f"Error: {e}")

    return {"frame": frame_num,
            "rooms":   answers[0],
            "doors":   answers[1],
            "windows": answers[2]}


# ── Phase B: JSON synthesis — llama3.2:3b (good at structured text output) ───

def _synthesize(observations, is962_context):
    """
    WHY llama3.2:3b here and NOT Moondream:
    - Moondream is a vision model — returns empty on long text-only prompts
    - llama3.2:3b is a text model — reliable structured JSON output
    - Phase A (image work) = Moondream, Phase B (text work) = llama3.2:3b
    """

    # Truncate IS 962 context — only keep the first 600 chars
    # (the key rules about A4, line types, lettering are in the first section)
    is962_short = is962_context[:600]

    # Cap to 5 frames maximum — avoids overwhelming the prompt
    obs_subset = observations[:5]

    # Build a compact summary — truncate each observation to avoid long prompts
    summary = ""
    for obs in obs_subset:
        summary += (
            f"\nFrame {obs['frame']}:\n"
            f"  Rooms  : {obs['rooms'][:150]}\n"
            f"  Doors  : {obs['doors'][:150]}\n"
            f"  Windows: {obs['windows'][:150]}\n"
        )

    prompt = f"""You are an architectural assistant creating a floor plan JSON.

Room observations from video frames:
{summary}

IS 962 rules (key points):
{is962_short}

Output ONLY a valid JSON object. No explanation, no markdown, just JSON:

{{
  "metadata": {{"title": "FLOOR PLAN", "scale": "NTS", "sheet": "A4", "orientation": "landscape"}},
  "rooms": [
    {{"id": "R1", "name": "Living Room", "x": 20, "y": 50, "width": 80, "height": 60}}
  ],
  "doors": [
    {{"room_id": "R1", "wall": "bottom", "position_ratio": 0.5, "width": 12, "swing": "left"}}
  ],
  "windows": [
    {{"room_id": "R1", "wall": "right", "position_ratio": 0.3, "width": 15}}
  ]
}}

Coordinate rules (mm, A4 landscape 297x210):
- Place rooms within x: 20-270, y: 35-185
- Size rooms proportionally (larger room = larger width/height)
- wall: "bottom" "top" "left" "right"
- position_ratio: 0.0 to 1.0 (where along that wall)
- swing: "left" or "right"

JSON only:"""

    for attempt in range(1, 4):
        try:
            resp = ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}  # Low temperature = consistent JSON
            )
            raw = resp["message"]["content"].strip()

            # Strip markdown fences if present
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

            # Extract only the JSON object (ignore any surrounding text)
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
    print("  Tip: manually edit output/floor_plan.json with room details.")
    return _fallback()


def _fallback():
    return {
        "metadata": {"title": "FLOOR PLAN", "scale": "NTS",
                     "sheet": "A4", "orientation": "landscape"},
        "rooms":   [{"id": "R1", "name": "Room",
                     "x": 40, "y": 60, "width": 200, "height": 120}],
        "doors":   [{"room_id": "R1", "wall": "bottom",
                     "position_ratio": 0.5, "width": 12, "swing": "left"}],
        "windows": [{"room_id": "R1", "wall": "right",
                     "position_ratio": 0.3, "width": 15}]
    }


# ── Public entry point ────────────────────────────────────────────────────────

def process_frames_with_vlm(frames_folder, is962_context):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    total = len(frame_files)

    # Cap at 8 frames — anything more overwhelms both Moondream and the prompt
    if total > 8:
        step = total // 8
        frame_files = [frame_files[i] for i in range(0, total, step)][:8]
        total = len(frame_files)
        print(f"  Capped to {total} evenly-spaced frames (was {len(os.listdir(frames_folder))} total)")

    print(f"  Phase A: Analyzing {total} frames with Moondream (image model)...")

    # Phase A — per-frame visual analysis with Moondream
    observations = []
    for i, fname in enumerate(frame_files):
        obs = _analyze_frame(os.path.join(frames_folder, fname), i + 1, total)
        observations.append(obs)
        print(f"    Rooms seen: {obs['rooms'][:80]}...")

    # Phase B — JSON synthesis with llama3.2:3b (text model)
    print("\n  Phase B: Synthesizing JSON with llama3.2:3b (text model)...")
    return _synthesize(observations, is962_context)


if __name__ == "__main__":
    from step0_parse_is_962 import extract_is962_context
    ctx  = extract_is962_context("input/IS 962.pdf")
    data = process_frames_with_vlm("output/frames", ctx)
    os.makedirs("output", exist_ok=True)
    with open("output/floor_plan.json", "w") as f:
        json.dump(data, f, indent=2)
    print("JSON saved: output/floor_plan.json")
    print(json.dumps(data, indent=2))