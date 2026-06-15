"""
Stage 4 — Global floor-plan structuring via Llama 3.2:3b (Ollama).

Changes from v1
---------------
* Added _slim_segments(): strips frame-level noise (frame_ids arrays, raw
  motion vectors, per-frame spatial characteristics) before the LLM call.
  A 3B model's context window fills quickly when full segment dicts are
  serialised; sending only the fields that drive floor-plan layout decisions
  (segment_id, room_type, size_hint, door_locations, confidence) measurably
  improves structured JSON output quality from small models.

Feeds the stage-3 segment summaries and transitions to Llama and asks it
to produce a single coherent JSON floor-plan spec.  If Ollama is not
reachable the fallback generator builds the spec directly from the
segment data without any LLM call.
"""
import json
import re
import requests
from typing import Optional

from pipeline.utils import SIZE_HINT_DIMS, assign_room_positions, compute_camera_path

_OLLAMA_URL  = "http://localhost:11434/api/generate"
_MODEL       = "llama3.2:3b"
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
6. Output ONLY valid JSON — no markdown, no extra text.

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


# ── public API ───────────────────────────────────────────────────────────────

def generate_floor_plan(segments: list, transitions: list) -> dict:
    """
    Call Llama via Ollama to produce the global floor-plan JSON.
    Falls back to a deterministic generator if Ollama is unavailable or
    returns invalid JSON.
    """
    prompt = _PROMPT_TEMPLATE.format(
        segments_json=json.dumps(_slim_segments(segments), indent=2),
        transitions_json=json.dumps(transitions, indent=2),
    )

    print("  Calling Llama 3.2:3b via Ollama …")
    raw = _call_ollama(prompt)

    if raw is None:
        print("  Ollama unreachable — using deterministic fallback.")
        return _fallback_floor_plan(segments, transitions)

    plan = _parse_and_validate(raw, segments, transitions)
    return plan


# ── segment slimming ─────────────────────────────────────────────────────────

def _slim_segments(segments: list) -> list:
    """
    Strip frame-level detail from segments before the LLM call.

    Fields removed
    --------------
    frame_ids            — can be a long list; useless for layout decisions
    segment_motion       — raw motion vectors; not needed by the LLM
    spatial_characteristics — per-frame text; inflates context without value
    is_boundary_heuristic   — pipeline-internal flag; LLM does not need it

    Fields kept
    -----------
    segment_id       — lets the LLM reference segments in its output
    room_type        — the primary layout signal
    size_hint        — drives room dimensions
    door_locations   — drives adjacency and door placement
    confidence       — lets the LLM weight ambiguous segments lower
    """
    return [
        {
            "segment_id":    s["segment_id"],
            "room_type":     s["room_type"],
            "size_hint":     s["size_hint"],
            "door_locations": s.get("door_locations", []),
            "confidence":    s["confidence"],
        }
        for s in segments
    ]


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

def _parse_and_validate(raw: str, segments: list, transitions: list) -> dict:
    """Extract JSON from LLM response; fall back if malformed."""
    start = raw.find('{')
    end   = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        print("  No JSON object in LLM output — using fallback.")
        return _fallback_floor_plan(segments, transitions)

    try:
        plan = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        print(f"  JSON parse error ({exc}) — using fallback.")
        return _fallback_floor_plan(segments, transitions)

    rooms = plan.get("rooms", [])
    if not rooms:
        return _fallback_floor_plan(segments, transitions)

    # Ensure rooms have correct IDs and dimensions
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

    # Filter transitions to those referencing valid room IDs
    valid_transitions = [
        t for t in plan.get("transitions", [])
        if t.get("from_room") in room_ids and t.get("to_room") in room_ids
    ]

    # Recompute camera_path from positions (LLM coords are unreliable at 3B scale)
    position_map = assign_room_positions(rooms, valid_transitions)
    camera_path  = compute_camera_path(rooms, valid_transitions, position_map)

    return {
        "rooms":       rooms,
        "transitions": valid_transitions,
        "camera_path": camera_path,
    }


# ── deterministic fallback ───────────────────────────────────────────────────

def _fallback_floor_plan(segments: list, transitions: list) -> dict:
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
        })
        room_id_by_type[rt] = rid

    # If everything was "unknown", create a single placeholder room
    if not rooms:
        rooms           = [{
            "id": "R0", "type": "unknown", "size_hint": "medium",
            "width": 4.0, "height": 4.0, "door_locations": [],
        }]
        room_id_by_type = {"unknown": "R0"}

    # Map stage-3 transitions to room IDs
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

    # Attach door_locations to rooms based on transitions
    _room_lookup = {r["id"]: r for r in rooms}
    for t in plan_transitions:
        room = _room_lookup.get(t["from_room"])
        if room is not None:
            room["door_locations"].append({
                "side":       t["door_position"],
                "to_room_id": t["to_room"],
            })

    position_map = assign_room_positions(rooms, plan_transitions)
    camera_path  = compute_camera_path(rooms, plan_transitions, position_map)

    return {
        "rooms":       rooms,
        "transitions": plan_transitions,
        "camera_path": camera_path,
    }