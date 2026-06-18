"""
Stage 4 — Global floor-plan structuring via Llama 3.2:3b (Ollama).

Changes from v3
---------------
* generate_floor_plan() now accepts an optional `frame_perception` list (raw
  Stage-2 output). This is forwarded to _attach_door_only_ghosts() which scans
  raw frame-level door observations for any door that:
    - was observed with confidence >= 0.45
    - has leads_to != "unknown"
    - is NOT already represented by an existing real or ghost room
  For each such door, a ghost room stub is created so the renderer can draw a
  dashed outline even when Stage 3 filtered the door out of segment data.

* _attach_untraversed_doors_and_ghosts() is unchanged — still creates ghosts
  from segment-level door data (the primary source).

* _slim_segments(): unchanged from v3.
"""
import json
import re
import requests
from typing import Optional

from pipeline.utils import SIZE_HINT_DIMS, assign_room_positions, compute_camera_path

_OLLAMA_URL  = "http://localhost:11434/api/generate"
_MODEL       = "qwen2.5:7b"
_TIMEOUT_SEC = 300

_PROMPT_TEMPLATE = """\
You are given semantic descriptions of indoor video segments and room transitions.
Produce a single JSON floor-plan spec.

Rules:
1. Create exactly one room entry per unique room type from the segments.
2. Do NOT invent rooms not mentioned in the input.
3. Assign dimensions from size_hint: very_small=2x2, small=3x3, medium=4x4, large=5x5, very_large=6x6.
4. Place rooms adjacently based on door_position: right→+x, left→-x, front→+y, back→-y.
5. camera_path visits room centres in transition order; heading_deg: 0=forward, 90=turn right, -90=turn left.
6. Use objects_visible and view_descriptions to resolve ambiguous room_type labels.
7. Output ONLY valid JSON — no markdown, no extra text.

Input segments (JSON array):
{segments_json}

Input transitions (JSON array):
{transitions_json}

Required output schema:
{{
  "rooms": [
    {{
      "id": "R0",
      "type": "<room type>",
      "size_hint": "<size hint>",
      "width": <number>,
      "height": <number>,
      "door_locations": [
        {{"side": "<left|right|front|back>", "to_room_id": "<room id>"}}
      ]
    }}
  ],
  "transitions": [
    {{
      "detected": <true|false>,
      "from_room": "<room id>",
      "to_room": "<room id>",
      "door_position": "<left|right|front|back>",
      "confidence": <0.0-1.0>
    }}
  ],
  "camera_path": [
    {{
      "x": <number>, "y": <number>,
      "heading_deg": <number>,
      "from_segment_id": <number>, "to_segment_id": <number>
    }}
  ]
}}"""


# ── room type → sensible ghost size ─────────────────────────────────────────

_GHOST_SIZE_BY_TYPE: dict[str, str] = {
    "bathroom":    "very_small",
    "office":      "small",
    "bedroom":     "medium",
    "kitchen":     "small",
    "entrance":    "small",
    "corridor":    "small",
    "living_room": "medium",
    "stairwell":   "small",
    "lobby":       "medium",
    "unknown":     "small",
}

# Minimum confidence for a raw-frame door to generate a ghost room
_GHOST_MIN_CONF = 0.45


# ── public API ───────────────────────────────────────────────────────────────

def generate_floor_plan(segments: list, transitions: list,
                        frame_perception: list | None = None) -> dict:
    """
    Call Llama via Ollama to produce the global floor-plan JSON.
    Falls back to a deterministic generator if Ollama is unavailable or
    returns invalid JSON.

    Parameters
    ----------
    segments        : Stage-3 segment list
    transitions     : Stage-3 transition list
    frame_perception: Stage-2 raw perception list (optional; used to create
                      ghost rooms for doors that Stage-3 may have filtered out)
    """
    prompt = _PROMPT_TEMPLATE.format(
        segments_json=json.dumps(_slim_segments(segments), indent=2),
        transitions_json=json.dumps(transitions, indent=2),
    )

    print("  Calling Llama 3.2:3b via Ollama …")
    raw = _call_ollama(prompt)

    if raw is None:
        print("  Ollama unreachable — using deterministic fallback.")
        return _fallback_floor_plan(segments, transitions, frame_perception)

    plan = _parse_and_validate(raw, segments, transitions, frame_perception)
    return plan


# ── segment slimming ─────────────────────────────────────────────────────────

def _slim_segments(segments: list) -> list:
    """
    Strip frame-level detail from segments before the LLM call.
    Includes to_room_type from the new rich door schema.
    """
    slimmed = []
    for s in segments:
        slim_doors = []
        for d in s.get("door_locations", []):
            slim_doors.append({
                "side":             d["side"],
                "to_room_type":     d.get("to_room_type", "unknown"),
                "confidence":       d.get("confidence", 0.5),
                "likely_traversed": d.get("likely_traversed", False),
            })
        slimmed.append({
            "segment_id":      s["segment_id"],
            "room_type":       s["room_type"],
            "size_hint":       s["size_hint"],
            "door_locations":  slim_doors,
            "objects_visible": s.get("objects_visible", [])[:5],
            "view_descriptions": s.get("view_descriptions", [])[:2],
            "confidence":      s["confidence"],
        })
    return slimmed


# ── Ollama I/O ───────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, retries: int = 2) -> Optional[str]:
    for attempt in range(retries):
        try:
            resp = requests.post(
                _OLLAMA_URL,
                json={
                    "model":   _MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "format":  "json",
                    "options": {"temperature": 0.05, "num_predict": 2048},
                },
                timeout=_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.ConnectionError:
            print(f"  Ollama connection error (attempt {attempt+1}/{retries})")
        except Exception as exc:
            print(f"  Ollama error: {exc} (attempt {attempt+1}/{retries})")
    return None


# ── JSON parsing & validation ────────────────────────────────────────────────

def _parse_and_validate(raw: str, segments: list, transitions: list,
                        frame_perception: list | None) -> dict:
    """Extract JSON from LLM response; fall back if malformed."""
    start = raw.find('{')
    end   = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        print("  No JSON object in LLM output — using fallback.")
        return _fallback_floor_plan(segments, transitions, frame_perception)

    try:
        plan = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        print(f"  JSON parse error ({exc}) — using fallback.")
        return _fallback_floor_plan(segments, transitions, frame_perception)

    rooms = plan.get("rooms", [])
    if not rooms:
        return _fallback_floor_plan(segments, transitions, frame_perception)

    room_ids: set = set()
    for i, room in enumerate(rooms):
        if not room.get("id"):
            room["id"] = f"R{i}"
        room_ids.add(room["id"])

        sh = room.get("size_hint", "medium")
        if sh not in SIZE_HINT_DIMS:
            sh = "medium"
            room["size_hint"] = sh
        room["width"], room["height"] = SIZE_HINT_DIMS[sh]

        if "door_locations" not in room:
            room["door_locations"] = []

        room.setdefault("ghost", False)

    valid_transitions = [
        t for t in plan.get("transitions", [])
        if t.get("from_room") in room_ids and t.get("to_room") in room_ids
    ]

    # Rebuild door_locations from validated transitions
    room_lookup = {r["id"]: r for r in rooms}
    for r in rooms:
        r["door_locations"] = []
    for t in valid_transitions:
        room = room_lookup.get(t["from_room"])
        if room is not None:
            room["door_locations"].append({
                "side":       t["door_position"],
                "to_room_id": t["to_room"],
                "traversed":  True,
            })

    # Primary ghost creation: from segment-level untraversed doors
    _attach_untraversed_doors_and_ghosts(rooms, segments, valid_transitions)

    # Secondary ghost creation: from raw frame perceptions (catches filtered doors)
    if frame_perception:
        _attach_door_only_ghosts(rooms, valid_transitions, frame_perception)

    position_map, heading_map = assign_room_positions(rooms, valid_transitions)
    camera_path  = compute_camera_path(rooms, valid_transitions, position_map, heading_map)

    return {
        "rooms":       rooms,
        "transitions": valid_transitions,
        "camera_path": camera_path,
    }


# ── deterministic fallback ───────────────────────────────────────────────────

def _fallback_floor_plan(segments: list, transitions: list,
                         frame_perception: list | None = None) -> dict:
    """Build a floor plan directly from stage-3 data without LLM."""
    seen_types:      dict = {}
    rooms:           list = []
    room_id_by_type: dict = {}

    for seg in segments:
        rt = seg["room_type"]
        if rt == "unknown" or rt in seen_types:
            continue
        seen_types[rt] = True
        rid       = f"R{len(rooms)}"
        size_hint = seg.get("size_hint", "medium")
        if size_hint not in SIZE_HINT_DIMS:
            size_hint = "medium"
        w, h = SIZE_HINT_DIMS[size_hint]
        rooms.append({
            "id":             rid,
            "type":           rt,
            "size_hint":      size_hint,
            "width":          w,
            "height":         h,
            "door_locations": [],
            "ghost":          False,
        })
        room_id_by_type[rt] = rid

    if not rooms:
        rooms           = [{
            "id": "R0", "type": "unknown", "size_hint": "medium",
            "width": 4.0, "height": 4.0, "door_locations": [], "ghost": False,
        }]
        room_id_by_type = {"unknown": "R0"}

    plan_transitions: list = []
    for t in transitions:
        fr = room_id_by_type.get(t["from_room_type"])
        tr = room_id_by_type.get(t["to_room_type"])
        if fr and tr and fr != tr:
            plan_transitions.append({
                "detected":      t["detected"],
                "from_room":     fr,
                "to_room":       tr,
                "door_position": t.get("door_position", "front"),
                "confidence":    t.get("confidence", 0.5),
            })

    _room_lookup = {r["id"]: r for r in rooms}
    for t in plan_transitions:
        room = _room_lookup.get(t["from_room"])
        if room is not None:
            room["door_locations"].append({
                "side":       t["door_position"],
                "to_room_id": t["to_room"],
                "traversed":  True,
            })

    _attach_untraversed_doors_and_ghosts(rooms, segments, plan_transitions)

    if frame_perception:
        _attach_door_only_ghosts(rooms, plan_transitions, frame_perception)

    position_map, heading_map = assign_room_positions(rooms, plan_transitions)
    camera_path  = compute_camera_path(rooms, plan_transitions, position_map, heading_map)

    return {
        "rooms":       rooms,
        "transitions": plan_transitions,
        "camera_path": camera_path,
    }


# ── shared helper: untraversed doors + ghost rooms ───────────────────────────

def _attach_untraversed_doors_and_ghosts(rooms: list, segments: list,
                                          transitions: list) -> None:
    """
    For each real room:
      1. Find door sides the VLM observed that are NOT already covered by a
         traversed transition door.
      2. For each such door, add it to the room's door_locations as traversed=False.
      3. Create a "ghost" room stub representing the unvisited space beyond that door.
      4. Add a synthetic (ghost) transition so assign_room_positions() places
         the ghost room on the correct wall.

    Ghost rooms are rendered as dashed outlines in Stage 5.
    """
    seg_by_type = {s["room_type"]: s for s in segments}
    existing_room_ids = {r["id"] for r in rooms}

    ghost_rooms:       list[dict] = []
    ghost_transitions: list[dict] = []

    for room in rooms:
        if room.get("ghost", False):
            continue

        rtype = room.get("type", "unknown")
        rid   = room["id"]
        seg   = seg_by_type.get(rtype)
        if not seg:
            continue

        traversed_sides: set = {
            d["side"] for d in room.get("door_locations", [])
            if d.get("traversed", True)
        }

        seen_ghost_sides: set = set()

        for d in seg.get("door_locations", []):
            side = d["side"]
            if side == "none" or side in traversed_sides or side in seen_ghost_sides:
                continue

            # Add untraversed door marker to the real room
            room["door_locations"].append({
                "side":       side,
                "to_room_id": None,
                "traversed":  False,
                "confidence": d.get("confidence", 0.5),
            })
            seen_ghost_sides.add(side)

            # Build ghost room
            leads_to  = d.get("to_room_type", "unknown") or "unknown"
            ghost_sh  = _GHOST_SIZE_BY_TYPE.get(leads_to, "small")
            g_w, g_h  = SIZE_HINT_DIMS[ghost_sh]
            ghost_id  = f"GHOST_{side}_{rid}"

            if ghost_id in existing_room_ids:
                continue
            existing_room_ids.add(ghost_id)

            ghost_room = {
                "id":             ghost_id,
                "type":           leads_to,
                "size_hint":      ghost_sh,
                "width":          g_w,
                "height":         g_h,
                "door_locations": [],
                "ghost":          True,
            }
            ghost_rooms.append(ghost_room)

            for dl in room["door_locations"]:
                if dl.get("side") == side and dl.get("traversed") is False and dl.get("to_room_id") is None:
                    dl["to_room_id"] = ghost_id
                    break

            ghost_transitions.append({
                "detected":      False,
                "from_room":     rid,
                "to_room":       ghost_id,
                "door_position": side,
                "confidence":    d.get("confidence", 0.5),
                "ghost":         True,
            })

    rooms.extend(ghost_rooms)
    transitions.extend(ghost_transitions)


# ── secondary ghost creation: raw frame-level door observations ───────────────

def _attach_door_only_ghosts(rooms: list, transitions: list,
                              frame_perception: list) -> None:
    """
    Scan raw Stage-2 frame perceptions for door observations that:
      - have confidence >= _GHOST_MIN_CONF
      - have leads_to != "unknown"
      - are NOT already covered by an existing real or ghost room with the
        same leads_to type

    For each such door, create a ghost room stub so the renderer can draw a
    dashed outline even if Stage 3 filtered the door out of segment-level data.

    This is the "safety net" for doors glimpsed in only 1-2 frames that
    _DOOR_THRESHOLD still couldn't preserve even at 0.15.
    """
    # Build a set of already-covered room types (real + ghost)
    covered_types: set[str] = {r.get("type", "unknown") for r in rooms}
    existing_room_ids: set[str] = {r["id"] for r in rooms}

    # Find the first real room (to attach ghost transitions from)
    real_rooms = [r for r in rooms if not r.get("ghost", False)]
    if not real_rooms:
        return

    ghost_rooms:       list[dict] = []
    ghost_transitions: list[dict] = []

    # Aggregate door observations across all frames
    # key: (side, leads_to) → best confidence seen
    door_obs: dict[tuple, float] = {}
    for fp in frame_perception:
        for d in fp.get("doors_visible", []):
            side     = d.get("side", "none")
            leads_to = d.get("leads_to", "unknown")
            conf     = d.get("confidence", 0.0)
            if side == "none" or leads_to == "unknown":
                continue
            key = (side, leads_to)
            door_obs[key] = max(door_obs.get(key, 0.0), conf)

    for (side, leads_to), best_conf in door_obs.items():
        if best_conf < _GHOST_MIN_CONF:
            continue
        if leads_to in covered_types:
            continue  # already have a room of this type

        # Use the first real room as the source (best approximation)
        source_room = real_rooms[0]
        rid         = source_room["id"]
        ghost_id    = f"GHOST_RAW_{side}_{leads_to}"

        if ghost_id in existing_room_ids:
            continue
        existing_room_ids.add(ghost_id)
        covered_types.add(leads_to)

        ghost_sh = _GHOST_SIZE_BY_TYPE.get(leads_to, "small")
        g_w, g_h = SIZE_HINT_DIMS[ghost_sh]

        ghost_room = {
            "id":             ghost_id,
            "type":           leads_to,
            "size_hint":      ghost_sh,
            "width":          g_w,
            "height":         g_h,
            "door_locations": [],
            "ghost":          True,
        }
        ghost_rooms.append(ghost_room)

        # Add untraversed door entry to the source real room
        source_room["door_locations"].append({
            "side":       side,
            "to_room_id": ghost_id,
            "traversed":  False,
            "confidence": round(best_conf, 2),
        })

        ghost_transitions.append({
            "detected":      False,
            "from_room":     rid,
            "to_room":       ghost_id,
            "door_position": side,
            "confidence":    round(best_conf, 2),
            "ghost":         True,
        })

        print(f"  [ghost] Door-only sighting: side={side} leads_to={leads_to} "
              f"conf={best_conf:.2f} → created {ghost_id}")

    rooms.extend(ghost_rooms)
    transitions.extend(ghost_transitions)