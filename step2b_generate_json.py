import ollama
import json
import os

# Drawing area bounds (mm, A4 landscape coordinate space)
DRAW_X1, DRAW_X2 = 25, 270
DRAW_Y1, DRAW_Y2 = 40, 185
MIN_W, MIN_H = 55, 45


# ── Pass 1: extract room names from observations ──────────────────────────────

def _identify_rooms(summary):
    """
    Ask the LLM only to list which distinct rooms appear in the video.
    This is a simple classification task that small models handle reliably.
    Returns an ordered list like ["Hallway", "Living Room", "Kitchen"].
    """
    prompt = f"""Read these indoor video observations and list every distinct room type mentioned.

Rules:
- Use standard room names: Living Room, Kitchen, Bedroom, Bathroom, Hallway, Staircase, Dining Room, Study, etc.
- List them in the order they FIRST appear in the observations.
- Only include rooms that are clearly visible (not just briefly glimpsed).
- Do NOT include rooms that are not mentioned at all.

OBSERVATIONS:
{summary}

Output ONLY a JSON array of strings, nothing else.
Example: ["Hallway", "Living Room", "Kitchen"]
"""
    for attempt in range(1, 4):
        try:
            resp = ollama.chat(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0}
            )
            raw = resp["message"]["content"].strip()
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                rooms = json.loads(raw[start:end])
                if isinstance(rooms, list) and len(rooms) > 0:
                    print(f"  Identified rooms: {rooms}")
                    return rooms
        except Exception as e:
            print(f"  Room identification attempt {attempt} failed: {e}")
    # Fallback: parse room names manually from summary text
    return _parse_rooms_from_text(summary)


def _parse_rooms_from_text(summary):
    """Regex-free fallback: scan for known room keywords in observations."""
    keywords = [
        "living room", "kitchen", "bedroom", "bathroom", "hallway",
        "corridor", "staircase", "dining room", "study", "landing",
    ]
    seen, order = set(), []
    lower = summary.lower()
    for kw in keywords:
        if kw in lower and kw not in seen:
            seen.add(kw)
            order.append(kw.title())
    return order if order else ["Entry", "Room A", "Room B"]


# ── Pass 2: layout synthesis ──────────────────────────────────────────────────

def _synthesize(summary, is962_context, room_names):
    """
    Second pass: given the actual room names already extracted,
    ask the LLM to assign 2D coordinates.
    The schema example uses the real room names so the LLM adapts to them.
    """
    if len(is962_context) > 400:
        cut = is962_context.rfind('.', 0, 400)
        is962_short = is962_context[:cut + 1] if cut != -1 else is962_context[:400]
    else:
        is962_short = is962_context

    # Build schema example using the ACTUAL observed room names
    schema_rooms = []
    # Room 0 is the entry/first seen — place at bottom
    positions = _default_positions(room_names)
    for i, name in enumerate(room_names):
        px, py, pw, ph = positions[i]
        schema_rooms.append(
            f'    {{"id": "R{i+1}", "name": "{name}", "x": {px}, "y": {py}, "width": {pw}, "height": {ph}}}'
        )

    schema_rooms_str = ",\n".join(schema_rooms)

    # Build door examples between adjacent rooms
    door_examples = []
    if len(room_names) >= 2:
        door_examples.append(
            '    {"room_id": "R1", "wall": "top", "position_ratio": 0.5, "width": 14, "swing": "left"}'
        )
    if len(room_names) >= 3:
        door_examples.append(
            '    {"room_id": "R1", "wall": "left", "position_ratio": 0.5, "width": 14, "swing": "right"}'
        )
    doors_str = ",\n".join(door_examples) if door_examples else \
        '    {"room_id": "R1", "wall": "bottom", "position_ratio": 0.5, "width": 14, "swing": "left"}'

    schema = f"""{{
  "metadata": {{
    "title": "HOME FLOOR PLAN",
    "units": "mm",
    "standard": "IS962",
    "confidence": 0.8
  }},
  "rooms": [
{schema_rooms_str}
  ],
  "doors": [
{doors_str}
  ],
  "windows": [
    {{"room_id": "R1", "wall": "top", "position_ratio": 0.3, "width": 20}}
  ],
  "stairs": [],
  "corridors": [],
  "adjacency_graph": {{}}
}}"""

    room_list_str = ", ".join(f'"{r}"' for r in room_names)

    prompt = f"""You are a floor plan layout engine. Your job is to assign 2D positions to a set of rooms observed in a video walkthrough.

THE ROOMS TO LAYOUT (use EXACTLY these names, no others):
[{room_list_str}]

CANVAS RULES (violating these will break the drawing):
- Drawing area: x = 20 to 275, y = 40 to 185. ALL rooms MUST stay fully inside.
- y increases upward: first room seen goes at BOTTOM (y ≈ 45). Rooms visited later go ABOVE (higher y).
- Rooms connected by a door MUST share an edge — their rectangle coordinates must touch.
- Do NOT add rooms that are not in the list above.

ROOM SIZING GUIDE (minimum sizes):
- Hallway / Corridor : width = 60-80,  height = 50-65
- Kitchen            : width = 70-95,  height = 65-85
- Living Room        : width = 95-135, height = 70-90
- Bedroom            : width = 80-105, height = 68-88
- Bathroom           : width = 45-60,  height = 45-60
- Staircase / Landing: width = 55-75,  height = 50-65

PLACEMENT:
1. First room in the list → bottom center: x ≈ 105, y = 45.
2. Each subsequent room → above the previous: y = previous_y + previous_height.
3. Side-by-side rooms → same y, adjust x so they touch.
4. Touching rule: vertically adjacent rooms share the same top/bottom y. Horizontally adjacent share the same left/right x.

DOORS: Place a door on the wall facing the connected room.
STAIRS: If "Staircase" or "Hallway" is in the room list and stairs were observed, add a stairs entry with room_id pointing to that room.

OBSERVATIONS FOR CONTEXT:
{summary}

Output ONLY valid JSON matching this schema structure. Use the exact room names from the list above:
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

            # Verify the room names match what we asked for
            got_names = {r.get("name", "") for r in data.get("rooms", [])}
            expected = set(room_names)
            if not got_names.issubset(expected | {n.upper() for n in expected}):
                print(f"  Warning: LLM used unexpected rooms {got_names - expected}, fixing...")
                data = _force_room_names(data, room_names)

            print(f"  JSON ready (attempt {attempt})")
            return data

        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < 3:
                print("  Retrying...")

    print("  WARNING: Could not produce valid JSON. Using computed fallback.")
    return _computed_fallback(room_names)


def _default_positions(room_names):
    """
    Generate reasonable starting coordinates for each room based on its type.
    These are starting hints; the post-processor will scale them to fit.
    """
    size_map = {
        "living room":  (105, None, 120, 80),
        "kitchen":      (30,  None, 80,  75),
        "bedroom":      (30,  None, 90,  78),
        "bathroom":     (30,  None, 50,  50),
        "hallway":      (105, None, 70,  55),
        "corridor":     (105, None, 65,  50),
        "staircase":    (105, None, 65,  55),
        "landing":      (105, None, 65,  55),
        "dining room":  (195, None, 80,  70),
        "study":        (195, None, 75,  68),
    }
    default = (105, None, 80, 70)

    # Stack rooms vertically: entry room at bottom, rest go up
    positions = []
    y_cursor = 45
    x_cursor = 105

    for i, name in enumerate(room_names):
        key = name.lower()
        sx, _, sw, sh = size_map.get(key, default)
        if i == 0:
            # Entry room — bottom center
            positions.append((sx, y_cursor, sw, sh))
            y_cursor += sh
        else:
            # Subsequent rooms — stack upward, keep x from type hint
            positions.append((sx, y_cursor, sw, sh))
            # Don't advance y_cursor for side-by-side rooms (same y as previous)
            # Heuristic: if room is wider than previous, place beside it
            if i > 1:
                prev_x, prev_y, prev_w, prev_h = positions[i - 1]
                if prev_y == y_cursor:
                    pass  # already side-by-side
                else:
                    y_cursor += sh

    return positions


def _force_room_names(data, room_names):
    """If the LLM used wrong room names, remap them to the correct ones."""
    rooms = data.get("rooms", [])
    for i, room in enumerate(rooms):
        if i < len(room_names):
            room["name"] = room_names[i]
    return data


# ── Post-processing ───────────────────────────────────────────────────────────

def _validate_and_fix(data):
    """
    Scale and center layout to fill the drawing area.
    Fix room sizes and clip to bounds.
    Reposition stairs inside their room.
    """
    rooms = data.get("rooms", [])
    if not rooms:
        return data

    for r in rooms:
        r.setdefault("x", DRAW_X1 + 10)
        r.setdefault("y", DRAW_Y1 + 10)
        r["width"]  = max(MIN_W, r.get("width",  MIN_W))
        r["height"] = max(MIN_H, r.get("height", MIN_H))

    min_x = min(r["x"] for r in rooms)
    min_y = min(r["y"] for r in rooms)
    max_x = max(r["x"] + r["width"]  for r in rooms)
    max_y = max(r["y"] + r["height"] for r in rooms)

    layout_w = max_x - min_x
    layout_h = max_y - min_y

    margin = 5
    avail_w = (DRAW_X2 - DRAW_X1) - 2 * margin
    avail_h = (DRAW_Y2 - DRAW_Y1) - 2 * margin

    if layout_w > 0 and layout_h > 0:
        scale = min(avail_w * 0.92 / layout_w, avail_h * 0.92 / layout_h)
        scale = max(0.5, min(scale, 3.0))
    else:
        scale = 1.0

    scaled_w = layout_w * scale
    scaled_h = layout_h * scale
    offset_x = DRAW_X1 + margin + (avail_w - scaled_w) / 2
    offset_y = DRAW_Y1 + margin + (avail_h - scaled_h) / 2

    for r in rooms:
        r["x"]      = round(offset_x + (r["x"] - min_x) * scale)
        r["y"]      = round(offset_y + (r["y"] - min_y) * scale)
        r["width"]  = max(MIN_W, round(r["width"]  * scale))
        r["height"] = max(MIN_H, round(r["height"] * scale))
        r["x"] = max(DRAW_X1, min(r["x"], DRAW_X2 - r["width"]))
        r["y"] = max(DRAW_Y1, min(r["y"], DRAW_Y2 - r["height"]))

    room_map = {r["id"]: r for r in rooms}
    fixed_stairs = []
    for stair in data.get("stairs", []):
        rid = stair.get("room_id")
        if rid and rid in room_map:
            room = room_map[rid]
            sw = min(30, room["width"]  - 8)
            sh = min(20, room["height"] - 8)
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


def _computed_fallback(room_names):
    """Build a valid fallback purely from the identified room names."""
    rooms, doors = [], []
    y = 45
    x = 105
    for i, name in enumerate(room_names):
        key = name.lower()
        w = 120 if "living" in key else 80 if "kitchen" in key or "bedroom" in key else 70
        h = 80 if "living" in key else 75 if "bedroom" in key else 55
        rooms.append({"id": f"R{i+1}", "name": name, "x": x, "y": y, "width": w, "height": h})
        if i > 0:
            doors.append({
                "room_id": f"R{i+1}", "wall": "bottom",
                "position_ratio": 0.5, "width": 14, "swing": "left"
            })
        y += h

    return {
        "metadata": {"title": "HOME FLOOR PLAN", "units": "mm",
                     "standard": "IS962", "confidence": 0.3},
        "rooms": rooms,
        "doors": doors,
        "windows": [{"room_id": "R1", "wall": "top", "position_ratio": 0.4, "width": 20}],
        "stairs": [],
        "corridors": [],
        "adjacency_graph": {},
    }


# ── Entry point ───────────────────────────────────────────────────────────────

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

    print("\n  Phase B1: Identifying rooms from observations...")
    room_names = _identify_rooms(summary)

    print(f"\n  Phase B2: Laying out {len(room_names)} rooms with llama3.2:3b...")
    data = _synthesize(summary, ctx, room_names)
    data = _validate_and_fix(data)

    os.makedirs("output", exist_ok=True)
    with open("output/floor_plan.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("\n[OK] Floor plan JSON saved to output/floor_plan.json")
