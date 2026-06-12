import ollama
import json
import os

def _synthesize(summary, is962_context):
    # SAFER TRUNCATION: Find the last period (.) before the 600th character
    if len(is962_context) > 600:
        cut_point = is962_context.rfind('.', 0, 600)
        is962_short = is962_context[:cut_point + 1] if cut_point != -1 else is962_context[:600]
    else:
        is962_short = is962_context

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
1. Invent approximate, logical 2D numerical coordinates (x, y, width, height) for every room. The canvas is exactly width=297, height=210.
2. START AT BOTTOM CENTER: Place the first room at approximately x=120, y=40. Make average room sizes around width=50, height=50.
3. As the camera moves forward, build new rooms UPWARDS on the map (INCREASE the Y coordinate).
4. If the camera moves left, DECREASE the X coordinate. If it moves right, INCREASE the X coordinate.
5. IMPORTANT: Keep all X coordinates between 20 and 270. Keep all Y coordinates between 40 and 190.
6. Place doors on the appropriate walls (top, bottom, left, right) with a position_ratio between 0.0 and 1.0.

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

            # Robust markdown parsing logic
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


if __name__ == "__main__":
    import sys
    
    obs_path = "output/observations.txt"
    if not os.path.exists(obs_path):
        print(f"Error: {obs_path} not found. Please run step2a_extract_observations.py first.")
        sys.exit(1)
        
    context_path = "output/is962_context.txt"
    if not os.path.exists(context_path):
        print(f"Error: {context_path} not found. Please run step0_parse_is_962.py first.")
        sys.exit(1)
        
    with open(obs_path, "r", encoding="utf-8") as f:
        summary = f.read()

    with open(context_path, "r", encoding="utf-8") as f:
        ctx = f.read()
        
    print("\n  Phase B: Synthesizing JSON with llama3.2:3b (text model)...")
    data = _synthesize(summary, ctx)
    
    os.makedirs("output", exist_ok=True)
    with open("output/floor_plan.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print("\n✅ Floor plan JSON saved to output/floor_plan.json")
