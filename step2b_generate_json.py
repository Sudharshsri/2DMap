import ollama
import json
import os

# Drawing area bounds (mm, A4 landscape coordinate space)
DRAW_X1, DRAW_X2 = 25, 270
DRAW_Y1, DRAW_Y2 = 40, 185
MIN_W, MIN_H = 55, 45


def _synthesize(summary, is962_context):
    if len(is962_context) > 400:
        cut = is962_context.rfind('.', 0, 400)
        is962_short = is962_context[:cut + 1] if cut != -1 else is962_context[:400]
    else:
        is962_short = is962_context

    schema = """
{
  "metadata": {
    "title": "HOME FLOOR PLAN",
    "units": "mm",
    "standard": "IS962",
    "confidence": 0.8
  },
  "rooms": [
    {"id": "R1", "name": "Hallway",     "x": 108, "y": 45,  "width": 72,  "height": 58},
    {"id": "R2", "name": "Living Room", "x": 108, "y": 103, "width": 118, "height": 78},
    {"id": "R3", "name": "Kitchen",     "x": 30,  "y": 103, "width": 78,  "height": 78}
  ],
  "doors": [
    {"room_id": "R1", "wall": "top",    "position_ratio": 0.7, "width": 14, "swing": "left"},
    {"room_id": "R1", "wall": "left",   "position_ratio": 0.5, "width": 14, "swing": "right"},
    {"room_id": "R1", "wall": "bottom", "position_ratio": 0.5, "width": 14, "swing": "left"}
  ],
  "windows": [
    {"room_id": "R2", "wall": "top",  "position_ratio": 0.3, "width": 25},
    {"room_id": "R3", "wall": "left", "position_ratio": 0.5, "width": 20}
  ],
  "stairs": [
    {"room_id": "R1", "x": 113, "y": 50, "width": 22, "height": 16}
  ],
  "corridors": [],
  "adjacency_graph": {"R1": ["R2", "R3"], "R2": ["R1"], "R3": ["R1"]}
}
"""

    prompt = f"""You are a floor plan layout engine. Analyze video walkthrough observations and produce a 2D floor plan JSON.

CANVAS RULES (CRITICAL — violating these will break the drawing):
- Drawing area: x = 20 to 275, y = 40 to 185. ALL rooms MUST stay fully within these bounds.
- y increases upward: place the entry/hallway room at the BOTTOM (y ≈ 45). Rooms visited later go ABOVE (higher y).
- Adjacent rooms (connected by a door) MUST share a wall — their rectangle edges must be equal or differ by ≤ 3.
- Do NOT invent rooms that were not mentioned in the observations.

ROOM SIZING GUIDE (these are minimum acceptable sizes):
- Hallway / corridor : width = 60-80,  height = 50-65
- Kitchen            : width = 70-95,  height = 65-85
- Living Room        : width = 95-135, height = 70-90
- Bedroom            : width = 80-105, height = 68-88
- Bathroom           : width = 45-60,  height = 45-60
- Staircase / landing: width = 55-75,  height = 50-65

PLACEMENT ALGORITHM (follow this exactly):
1. Identify the first room seen in the observations — place it at bottom center: x ≈ 105, y = 45.
2. Each room entered after that goes ABOVE the previous room: new_y = previous_y + previous_height.
3. If the camera moved LEFT, decrease x by the new room's width (place it to the left).
4. If the camera moved RIGHT, keep or increase x (place it to the right).
5. Touching rule: when two rooms are vertically adjacent, their top/bottom y coordinates must match exactly.
   When horizontally adjacent, their left/right x coordinates must match exactly.

DOORS: door wall = the wall of the room facing the adjacent room.
STAIRS: if stairs were observed, add them with room_id = the room they appear in.

VIDEO OBSERVATIONS (read carefully — use ONLY rooms mentioned here):
{summary}

Adapt room names, count, and positions to match what is actually observed.
Output ONLY valid JSON matching this exact schema structure (no extra text, no markdown):
{schema}
"""

    for attempt in range(1, 4):
        try:
            resp = ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.05}
            )
            raw = resp["message"]["content"].strip()

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

            start = raw.find("{")
            end = raw.rfind("}") + 1
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


def _validate_and_fix(data):
    """
    Post-process LLM output to fix common layout errors:
      1. Enforce minimum room sizes.
      2. Scale and center the entire layout to fill the drawing area.
      3. Clip any remaining out-of-bounds rooms.
      4. Reposition stairs to be inside their referenced room.
    """
    rooms = data.get("rooms", [])
    if not rooms:
        return data

    # Step 1 — enforce minimum sizes and default positions
    for r in rooms:
        r.setdefault("x", DRAW_X1 + 10)
        r.setdefault("y", DRAW_Y1 + 10)
        r["width"]  = max(MIN_W, r.get("width", MIN_W))
        r["height"] = max(MIN_H, r.get("height", MIN_H))

    # Step 2 — compute bounding box of the LLM-produced layout
    min_x = min(r["x"] for r in rooms)
    min_y = min(r["y"] for r in rooms)
    max_x = max(r["x"] + r["width"]  for r in rooms)
    max_y = max(r["y"] + r["height"] for r in rooms)

    layout_w = max_x - min_x
    layout_h = max_y - min_y

    # Step 3 — scale to fill ~92 % of the available drawing area
    margin = 5
    avail_w = (DRAW_X2 - DRAW_X1) - 2 * margin
    avail_h = (DRAW_Y2 - DRAW_Y1) - 2 * margin

    target_w = avail_w * 0.92
    target_h = avail_h * 0.92

    if layout_w > 0 and layout_h > 0:
        scale = min(target_w / layout_w, target_h / layout_h)
        scale = max(0.5, min(scale, 3.0))   # clamp to sane range
    else:
        scale = 1.0

    # Step 4 — apply scale and center inside drawing area
    scaled_w = layout_w * scale
    scaled_h = layout_h * scale
    offset_x = DRAW_X1 + margin + (avail_w - scaled_w) / 2
    offset_y = DRAW_Y1 + margin + (avail_h - scaled_h) / 2

    for r in rooms:
        r["x"]      = round(offset_x + (r["x"] - min_x) * scale)
        r["y"]      = round(offset_y + (r["y"] - min_y) * scale)
        r["width"]  = max(MIN_W, round(r["width"]  * scale))
        r["height"] = max(MIN_H, round(r["height"] * scale))
        # hard clip to drawing area
        r["x"] = max(DRAW_X1, min(r["x"], DRAW_X2 - r["width"]))
        r["y"] = max(DRAW_Y1, min(r["y"], DRAW_Y2 - r["height"]))

    # Step 5 — reposition stairs to sit inside their referenced room
    room_map = {r["id"]: r for r in rooms}
    fixed_stairs = []
    for stair in data.get("stairs", []):
        rid = stair.get("room_id")
        if rid and rid in room_map:
            room = room_map[rid]
            sw = min(30, room["width"]  - 8)
            sh = min(20, room["height"] - 8)
            # Place stairs at the TOP of the room (nearest the rooms above it)
            fixed_stairs.append({
                "room_id": rid,
                "x":       room["x"] + 4,
                "y":       room["y"] + room["height"] - sh - 3,
                "width":   sw,
                "height":  sh,
            })
        elif "x" in stair and "y" in stair:
            fixed_stairs.append({
                "x":      round(offset_x + (stair["x"] - min_x) * scale),
                "y":      round(offset_y + (stair["y"] - min_y) * scale),
                "width":  stair.get("width",  20),
                "height": stair.get("height", 15),
            })
    data["stairs"] = fixed_stairs

    return data


def _fallback():
    return {
        "metadata": {
            "title": "HOME FLOOR PLAN",
            "units": "mm",
            "standard": "IS962",
            "confidence": 0.3,
        },
        "rooms": [
            {"id": "R1", "name": "Hallway",     "x": 108, "y": 45,  "width": 72,  "height": 58},
            {"id": "R2", "name": "Living Room", "x": 108, "y": 103, "width": 118, "height": 78},
            {"id": "R3", "name": "Kitchen",     "x": 30,  "y": 103, "width": 78,  "height": 78},
        ],
        "doors": [
            {"room_id": "R1", "wall": "top",    "position_ratio": 0.7, "width": 14, "swing": "left"},
            {"room_id": "R1", "wall": "left",   "position_ratio": 0.5, "width": 14, "swing": "right"},
            {"room_id": "R1", "wall": "bottom", "position_ratio": 0.5, "width": 14, "swing": "left"},
        ],
        "windows": [
            {"room_id": "R2", "wall": "top",  "position_ratio": 0.3, "width": 25},
            {"room_id": "R3", "wall": "left", "position_ratio": 0.5, "width": 20},
        ],
        "stairs": [
            {"room_id": "R1", "x": 113, "y": 50, "width": 22, "height": 16},
        ],
        "corridors": [],
        "adjacency_graph": {"R1": ["R2", "R3"], "R2": ["R1"], "R3": ["R1"]},
    }


if __name__ == "__main__":
    import sys

    obs_path = "output/observations.txt"
    if not os.path.exists(obs_path):
        print(f"Error: {obs_path} not found. Run step2a_extract_observations.py first.")
        sys.exit(1)

    context_path = "output/is962_context.txt"
    if not os.path.exists(context_path):
        print(f"Error: {context_path} not found. Run step0_parse_is_962.py first.")
        sys.exit(1)

    with open(obs_path, "r", encoding="utf-8") as f:
        summary = f.read()

    with open(context_path, "r", encoding="utf-8") as f:
        ctx = f.read()

    print("\n  Phase B: Synthesizing JSON with llama3.2:3b (text model)...")
    data = _synthesize(summary, ctx)
    data = _validate_and_fix(data)

    os.makedirs("output", exist_ok=True)
    with open("output/floor_plan.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("\n[OK] Floor plan JSON saved to output/floor_plan.json")
