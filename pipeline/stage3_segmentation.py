"""
Stage 3 — Frame-to-segment grouping and transition detection.

Changes from v1
---------------
* SEMANTIC_CANONICAL map applied as a secondary safety net (primary is Stage 2)
  so any synonym that slipped through the VLM parse is caught here too.
* _smooth_room_types(): temporal majority-vote smoothing over a configurable
  window before the grouping pass.  A single-frame label flip (e.g. frame 8
  "hallway" between two "corridor" frames) is corrected before it ever reaches
  the grouping loop, preventing spurious one-frame segments.
* _merge_short_segments(): post-grouping pass that absorbs segments shorter
  than _MIN_SEGMENT_FRAMES into their highest-confidence neighbour.  Acts as a
  final safety net for any residual short segments that survive smoothing.
* _build_segment() now uses weighted confidence voting for room_type instead
  of a simple mode count.  Low-confidence frames contribute partial weight to
  their alternative_type, so a single uncertain frame cannot flip the segment
  label but does leave a soft signal if many uncertain frames agree on an alt.
* is_boundary_heuristic is now derived from motion rotation data rather than
  from the VLM output.  Sharp camera rotation is a more reliable boundary
  signal than asking a 3B VLM to predict pipeline state.
"""
from collections import Counter

_SIZE_HINT_ORDER       = ["very_small", "small", "medium", "large", "very_large"]
_DOOR_THRESHOLD        = 0.30    # fraction of frames that must show a door side
_BOUNDARY_FRAC         = 0.20    # fraction of frames with high rotation = boundary
_ROTATION_BOUNDARY_DEG = 25.0   # degrees; above this counts as a boundary signal
_MIN_SEGMENT_FRAMES    = 3       # segments shorter than this get merged
_SMOOTH_WINDOW         = 3       # temporal smoothing window (odd number works best)

# Secondary semantic normalization — mirrors Stage 2's SEMANTIC_CANONICAL.
# Catches anything that slipped through (e.g. loaded from a cached stage2 JSON
# produced by the old code before hallway was removed from the vocabulary).
SEMANTIC_CANONICAL: dict[str, str] = {
    "hallway":    "corridor",
    "passage":    "corridor",
    "walkway":    "corridor",
    "passageway": "corridor",
}


def _normalize_room_type(rt: str) -> str:
    return SEMANTIC_CANONICAL.get(rt, rt)


# ── public API ───────────────────────────────────────────────────────────────

def segment_frames(frame_perception: list, frame_motion: list) -> list:
    """
    Group frames into segments by stable room_type.

    Pipeline
    --------
    1. Semantic normalisation  — collapse any synonym labels from Stage 2
    2. Temporal smoothing      — majority vote in a 3-frame window
    3. Grouping                — consecutive same-type runs become segments;
                                 'unknown' frames are absorbed into the
                                 current segment rather than splitting it
    4. Short-segment merging   — absorb segments < _MIN_SEGMENT_FRAMES frames
                                 into their highest-confidence neighbour
    """
    if not frame_perception:
        return []

    motion_by_id = {m["frame_id"]: m for m in frame_motion}

    # Step 1 — secondary semantic normalisation
    frame_perception = [
        {**fp, "room_type": _normalize_room_type(fp.get("room_type", "unknown"))}
        for fp in frame_perception
    ]

    # Step 2 — temporal smoothing
    frame_perception = _smooth_room_types(frame_perception, window=_SMOOTH_WINDOW)

    # Step 3 — group into segments
    segments: list[dict]  = []
    current_frames: list  = []
    current_type: str | None = None
    seg_id = 0

    for fp in frame_perception:
        rt = fp.get("room_type", "unknown")

        if current_type is None:
            current_type   = rt
            current_frames = [fp]
        elif rt == current_type or rt == "unknown":
            current_frames.append(fp)
        else:
            segments.append(
                _build_segment(seg_id, current_frames, current_type, motion_by_id)
            )
            seg_id        += 1
            current_type   = rt
            current_frames = [fp]

    if current_frames:
        segments.append(
            _build_segment(seg_id, current_frames,
                           current_type or "unknown", motion_by_id)
        )

    # Step 4 — merge short segments
    segments = _merge_short_segments(segments, motion_by_id)

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

        # Best door side from the "leaving" segment
        door_position = "front"
        if seg_a.get("door_locations"):
            best = max(seg_a["door_locations"], key=lambda d: d["confidence"])
            door_position = best["side"]

        confidence = round((seg_a["confidence"] + seg_b["confidence"]) / 2, 2)

        transitions.append({
            "detected":        detected,
            "from_room_type":  seg_a["room_type"],
            "to_room_type":    seg_b["room_type"],
            "from_segment_id": seg_a["segment_id"],
            "to_segment_id":   seg_b["segment_id"],
            "door_position":   door_position,
            "confidence":      confidence,
        })

    return transitions


# ── private helpers ──────────────────────────────────────────────────────────

def _smooth_room_types(perceptions: list, window: int = 3) -> list:
    """
    Replace each frame's room_type with the majority type in its local window.

    A frame's label is only overwritten when:
      (a) the frame is 'unknown', OR
      (b) a different label has a strict majority in the window.

    This means confident, consistent runs are never altered; only lone
    outliers (a single 'hallway' between two 'corridor' frames) are corrected.
    """
    n    = len(perceptions)
    half = window // 2
    smoothed = []

    for i, fp in enumerate(perceptions):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        # Collect non-unknown labels in the window
        window_types = [
            p["room_type"] for p in perceptions[lo:hi]
            if p["room_type"] != "unknown"
        ]

        if window_types:
            majority       = Counter(window_types).most_common(1)[0][0]
            current        = fp.get("room_type", "unknown")
            count_current  = window_types.count(current)
            count_majority = window_types.count(majority)

            # Overwrite only when current is unknown or genuinely outvoted
            if current == "unknown" or (majority != current
                                        and count_majority > count_current):
                fp = {**fp, "room_type": majority}

        smoothed.append(fp)

    return smoothed


def _merge_short_segments(segments: list, motion_by_id: dict) -> list:
    """
    Iteratively merge segments shorter than _MIN_SEGMENT_FRAMES into the
    neighbour with the higher confidence score.

    Only frame_ids are transferred; the absorbing segment's room_type and
    other properties are preserved (it is the dominant segment by definition).
    Segment IDs are re-numbered sequentially after all merges are done.
    """
    if len(segments) <= 1:
        return segments

    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            if len(seg["frame_ids"]) >= _MIN_SEGMENT_FRAMES:
                continue

            # Choose the highest-confidence neighbour
            if i == 0:
                target_idx = 1
            elif i == len(segments) - 1:
                target_idx = i - 1
            else:
                prev_conf = segments[i - 1]["confidence"]
                next_conf = segments[i + 1]["confidence"]
                target_idx = i - 1 if prev_conf >= next_conf else i + 1

            target = segments[target_idx]

            # Merge frame_ids in temporal order
            if target_idx > i:
                target["frame_ids"] = seg["frame_ids"] + target["frame_ids"]
            else:
                target["frame_ids"] = target["frame_ids"] + seg["frame_ids"]

            segments.pop(i)
            changed = True
            break  # restart scan — indices have shifted

    # Re-number segment_ids to be contiguous from 0
    for idx, seg in enumerate(segments):
        seg["segment_id"] = idx

    return segments


def _build_segment(segment_id: int, frames: list,
                   room_type_hint: str, motion_by_id: dict) -> dict:
    frame_ids = [f["frame_id"] for f in frames]
    n         = len(frames)

    # ── room_type: weighted confidence voting ─────────────────────────────────
    # Each frame contributes its full confidence to room_type and a fractional
    # credit to alternative_type when confidence is low.  This means a single
    # uncertain frame cannot flip the segment label, but many low-confidence
    # frames agreeing on an alternative can influence the result.
    room_type_votes: Counter = Counter()
    for f in frames:
        rt   = f.get("room_type", "unknown")
        conf = f.get("confidence", 0.5)
        alt  = f.get("alternative_type")

        if rt != "unknown":
            room_type_votes[rt] += conf

        # Partial credit: the more uncertain the frame, the more the alt counts
        if conf < 0.65 and alt and alt != "unknown":
            room_type_votes[alt] += (1.0 - conf) * 0.5

    room_type = (
        room_type_votes.most_common(1)[0][0]
        if room_type_votes else room_type_hint or "unknown"
    )

    # ── size_hint: mode ───────────────────────────────────────────────────────
    hints     = [f["size_hint"] for f in frames if f.get("size_hint") in _SIZE_HINT_ORDER]
    size_hint = Counter(hints).most_common(1)[0][0] if hints else "medium"

    # ── door_locations: threshold-based aggregation ───────────────────────────
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

    # ── boundary heuristic: derived from motion rotation ──────────────────────
    # A sharp camera rotation signals a doorway or turn — a far more reliable
    # boundary cue than asking the VLM to predict pipeline state.
    rotations      = [abs(motion_by_id.get(fid, {}).get("rotation_deg", 0.0))
                      for fid in frame_ids]
    high_rot_count = sum(1 for r in rotations if r >= _ROTATION_BOUNDARY_DEG)
    is_boundary    = (high_rot_count / n) >= _BOUNDARY_FRAC

    # ── motion aggregation ────────────────────────────────────────────────────
    motions        = [motion_by_id.get(fid, {}) for fid in frame_ids]
    directions     = [
        m.get("motion_direction", "unknown")
        for m in motions
        if m.get("motion_direction") not in (None, "unknown")
    ]
    all_rotations  = [m.get("rotation_deg", 0.0) for m in motions]

    seg_direction = Counter(directions).most_common(1)[0][0] if directions else "forward"
    seg_rotation  = round(sum(all_rotations) / len(all_rotations), 1) if all_rotations else 0.0

    # ── spatial characteristics: top-5 by frequency ──────────────────────────
    char_counter: Counter = Counter()
    for f in frames:
        for c in f.get("spatial_characteristics", []):
            char_counter[c] += 1
    spatial_chars = [c for c, _ in char_counter.most_common(5)]

    # ── average confidence ────────────────────────────────────────────────────
    confs          = [f.get("confidence", 0.5) for f in frames]
    avg_confidence = round(sum(confs) / len(confs), 2)

    return {
        "segment_id":              segment_id,
        "room_type":               room_type,
        "size_hint":               size_hint,
        "door_locations":          door_locations,
        "is_boundary_heuristic":   is_boundary,
        "spatial_characteristics": spatial_chars,
        "segment_motion": {
            "direction":    seg_direction,
            "rotation_deg": seg_rotation,
        },
        "frame_ids":   frame_ids,
        "confidence":  avg_confidence,
    }