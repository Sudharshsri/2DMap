"""
Stage 3 — Frame-to-segment grouping and transition detection.

Groups consecutive frames with the same room_type into segments,
aggregates per-segment properties, then detects room transitions.
"""
from collections import Counter

_SIZE_HINT_ORDER  = ["very_small", "small", "medium", "large", "very_large"]
_DOOR_THRESHOLD   = 0.30   # fraction of frames that must show a door side
_BOUNDARY_FRAC    = 0.20   # fraction of frames flagged as boundary


# ── public API ───────────────────────────────────────────────────────────────

def segment_frames(frame_perception: list, frame_motion: list) -> list:
    """
    Group frames into segments by stable room_type.

    'unknown' frames are absorbed into the current segment rather than
    triggering a new one, so short gaps don't fracture the grouping.
    """
    if not frame_perception:
        return []

    motion_by_id = {m["frame_id"]: m for m in frame_motion}

    segments: list[dict]  = []
    current_frames: list  = []
    current_type: str | None = None
    seg_id = 0

    for fp in frame_perception:
        rt = fp.get("room_type", "unknown")

        if current_type is None:
            current_type = rt
            current_frames = [fp]
        elif rt == current_type or rt == "unknown":
            current_frames.append(fp)
        else:
            segments.append(
                _build_segment(seg_id, current_frames, current_type, motion_by_id)
            )
            seg_id       += 1
            current_type  = rt
            current_frames = [fp]

    if current_frames:
        segments.append(
            _build_segment(seg_id, current_frames, current_type or "unknown",
                           motion_by_id)
        )

    return segments


def detect_transitions(segments: list) -> list:
    """
    Identify room-to-room transitions between consecutive segments.

    A transition is *detected* when:
      - room types differ, AND
      - at least one segment has boundary heuristic OR visible doors.
    """
    transitions: list[dict] = []

    for i in range(len(segments) - 1):
        seg_a = segments[i]
        seg_b = segments[i + 1]

        if seg_a["room_type"] == seg_b["room_type"]:
            continue

        a_boundary = seg_a["is_boundary_heuristic"]
        b_boundary = seg_b["is_boundary_heuristic"]
        a_has_door = any(
            d["side"] != "none" for d in seg_a.get("door_locations", [])
        )
        detected = a_boundary or b_boundary or a_has_door

        # Choose best door side from the "leaving" segment
        door_position = "front"
        if seg_a.get("door_locations"):
            best = max(seg_a["door_locations"], key=lambda d: d["confidence"])
            door_position = best["side"]

        confidence = round((seg_a["confidence"] + seg_b["confidence"]) / 2, 2)

        transitions.append({
            "detected":          detected,
            "from_room_type":    seg_a["room_type"],
            "to_room_type":      seg_b["room_type"],
            "from_segment_id":   seg_a["segment_id"],
            "to_segment_id":     seg_b["segment_id"],
            "door_position":     door_position,
            "confidence":        confidence,
        })

    return transitions


# ── private helpers ──────────────────────────────────────────────────────────

def _build_segment(segment_id: int, frames: list,
                   room_type_hint: str, motion_by_id: dict) -> dict:
    frame_ids = [f["frame_id"] for f in frames]
    n         = len(frames)

    # --- room_type: mode excluding "unknown" ---
    rts  = [f["room_type"] for f in frames if f["room_type"] != "unknown"]
    room_type = (Counter(rts).most_common(1)[0][0]
                 if rts else room_type_hint or "unknown")

    # --- size_hint: mode ---
    hints     = [f["size_hint"] for f in frames if f.get("size_hint") in _SIZE_HINT_ORDER]
    size_hint = Counter(hints).most_common(1)[0][0] if hints else "medium"

    # --- door_locations: threshold-based aggregation ---
    door_counts: Counter = Counter()
    for f in frames:
        for d in f.get("doors_visible", ["none"]):
            door_counts[d] += 1

    door_locations = [
        {
            "side":         side,
            "to_room_type": "unknown",
            "confidence":   round(count / n, 2),
        }
        for side, count in door_counts.items()
        if side != "none" and count / n >= _DOOR_THRESHOLD
    ]

    # --- boundary heuristic ---
    boundary_count = sum(
        1 for f in frames if f.get("is_boundary_heuristic")
    )
    is_boundary = boundary_count / n >= _BOUNDARY_FRAC

    # --- motion aggregation ---
    motions    = [motion_by_id.get(fid, {}) for fid in frame_ids]
    directions = [
        m.get("motion_direction", "unknown")
        for m in motions
        if m.get("motion_direction") not in (None, "unknown")
    ]
    rotations  = [m.get("rotation_deg", 0.0) for m in motions]

    seg_direction = (Counter(directions).most_common(1)[0][0]
                     if directions else "forward")
    seg_rotation  = round(sum(rotations) / len(rotations), 1) if rotations else 0.0

    # --- spatial characteristics: top-5 by frequency ---
    char_counter: Counter = Counter()
    for f in frames:
        for c in f.get("spatial_characteristics", []):
            char_counter[c] += 1
    spatial_chars = [c for c, _ in char_counter.most_common(5)]

    # --- average confidence ---
    confs          = [f.get("confidence", 0.5) for f in frames]
    avg_confidence = round(sum(confs) / len(confs), 2)

    return {
        "segment_id":            segment_id,
        "room_type":             room_type,
        "size_hint":             size_hint,
        "door_locations":        door_locations,
        "is_boundary_heuristic": is_boundary,
        "spatial_characteristics": spatial_chars,
        "segment_motion": {
            "direction":    seg_direction,
            "rotation_deg": seg_rotation,
        },
        "frame_ids":  frame_ids,
        "confidence": avg_confidence,
    }
