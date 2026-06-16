"""
Stage 3 — Frame-to-segment grouping and transition detection.

Changes from v2
---------------
* _smooth_room_types(): now respects significant_change boundaries from Stage 2.
  A frame flagged significant_change=True is never smoothed away, and the
  look-behind window does not cross such a frame for its neighbours either.
  This prevents smoothing from reaching across a genuine room transition.

* _build_segment() boundary heuristic now combines two independent signals:
    rotation_boundary — Stage 1 optical-flow rotation (existing)
    vlm_boundary      — fraction of frames where Stage 2 flagged significant_change
  Either signal alone is sufficient to mark a segment as a boundary.

* _build_segment() aggregates two new Stage 2 fields:
    objects_visible   — top-10 objects by cross-frame frequency
    view_descriptions — deduplicated list of per-frame view_description strings
  Both are stored on the segment dict for Stage 4 context.

* segment_motion gains a new vlm_direction field derived from the VLM's
  camera_movement labels (Stage 2 changes_from_previous), complementing the
  optical-flow direction from Stage 1.
"""
from collections import Counter

_SIZE_HINT_ORDER       = ["very_small", "small", "medium", "large", "very_large"]
_DOOR_THRESHOLD        = 0.30    # fraction of frames that must show a door side
_BOUNDARY_FRAC         = 0.20    # fraction of frames with high rotation = boundary
_VLM_BOUNDARY_FRAC     = 0.25    # fraction of frames with significant_change = boundary
_ROTATION_BOUNDARY_DEG = 25.0    # degrees; above this counts as a boundary signal
_MIN_SEGMENT_FRAMES    = 2       # segments shorter than this get merged
_SMOOTH_WINDOW         = 3       # temporal smoothing window (odd number works best)

# Secondary semantic normalization — mirrors Stage 2's SEMANTIC_CANONICAL.
# Catches synonyms that slipped through (e.g. loaded from a cached stage2 JSON
# produced before the vocabulary was locked down).
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
    1. Semantic normalisation  — collapse synonym labels from Stage 2
    2. Temporal smoothing      — majority vote in a 3-frame window;
                                 significant_change frames act as hard stops
    3. Grouping                — consecutive same-type runs become segments;
                                 'unknown' frames are absorbed into the current
                                 segment rather than splitting it
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

    # Step 2 — temporal smoothing (respects significant_change boundaries)
    frame_perception = _smooth_room_types(frame_perception, window=_SMOOTH_WINDOW)

    # Step 3 — group into segments
    segments: list[dict]     = []
    current_frames: list     = []
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

    A transition is *detected* when room types differ AND any of:
      - seg_a or seg_b has is_boundary_heuristic=True  (rotation OR vlm_significant_change)
      - seg_a has visible door locations
      - the first frame of seg_b flagged significant_change=True  (direct VLM evidence
        of a scene shift at the exact boundary between the two segments)
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
        b_has_door = any(
            d["side"] != "none" for d in seg_b.get("door_locations", [])
        )
        detected = a_boundary or b_boundary or a_has_door or b_has_door

        # Best door side: prefer seg_a (leaving side), fall back to seg_b (entering side)
        door_position = "front"
        if seg_a.get("door_locations"):
            best = max(seg_a["door_locations"], key=lambda d: d["confidence"])
            door_position = best["side"]
        elif seg_b.get("door_locations"):
            best = max(seg_b["door_locations"], key=lambda d: d["confidence"])
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

    Boundary-awareness rules
    ------------------------
    1. A frame with significant_change=True is never overwritten — it marks a
       genuine scene transition and its label should stand as-is.
    2. When building the smoothing window for frame i, any look-behind frame
       that has significant_change=True is skipped.  This prevents the majority
       vote from reaching across a transition boundary.

    Normal smoothing rules (unchanged from v2)
    ------------------------------------------
    A frame's label is only overwritten when:
      (a) the frame is 'unknown', OR
      (b) a different label has a strict majority in the window.
    Confident, consistent runs are never altered; only lone outliers are
    corrected (e.g. a single 'kitchen' between two 'living_room' frames).
    """
    n    = len(perceptions)
    half = window // 2
    smoothed = []

    for i, fp in enumerate(perceptions):
        # The first frame has no previous-frame comparison context (sig is always False
        # for it by design).  Never smooth it — its label is the best evidence we have
        # for the room where the walk starts.
        if i == 0:
            smoothed.append(fp)
            continue

        # Rule 1: significant-change frames are immune to smoothing
        if fp.get("changes_from_previous", {}).get("significant_change", False):
            smoothed.append(fp)
            continue

        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        # Rule 2: build the window, skipping look-behind significant-change frames
        window_types = []
        for j in range(lo, hi):
            p = perceptions[j]
            if j < i and p.get("changes_from_previous", {}).get("significant_change", False):
                # Stop look-behind at this boundary — don't include it or anything before it
                continue
            if p["room_type"] != "unknown":
                window_types.append(p["room_type"])

        if window_types:
            majority       = Counter(window_types).most_common(1)[0][0]
            current        = fp.get("room_type", "unknown")
            count_current  = window_types.count(current)
            count_majority = window_types.count(majority)

            if current == "unknown" or (majority != current
                                        and count_majority > count_current):
                fp = {**fp, "room_type": majority}

        smoothed.append(fp)

    return smoothed


def _merge_short_segments(segments: list, motion_by_id: dict) -> list:
    """
    Iteratively merge segments shorter than _MIN_SEGMENT_FRAMES into the
    neighbour with the higher confidence score.

    Only frame_ids are transferred; the absorbing segment's room_type and other
    properties are preserved.  Segment IDs are re-numbered after all merges.
    """
    if len(segments) <= 1:
        return segments

    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            if len(seg["frame_ids"]) >= _MIN_SEGMENT_FRAMES:
                continue

            # Never merge the first or last segment — they represent where the
            # walkthrough starts and ends, so even a single frame is meaningful.
            if i == 0 or i == len(segments) - 1:
                continue

            if i == 0:
                target_idx = 1
            elif i == len(segments) - 1:
                target_idx = i - 1
            else:
                prev_conf = segments[i - 1]["confidence"]
                next_conf = segments[i + 1]["confidence"]
                target_idx = i - 1 if prev_conf >= next_conf else i + 1

            target = segments[target_idx]

            if target_idx > i:
                target["frame_ids"] = seg["frame_ids"] + target["frame_ids"]
            else:
                target["frame_ids"] = target["frame_ids"] + seg["frame_ids"]

            segments.pop(i)
            changed = True
            break  # restart scan — indices have shifted

    for idx, seg in enumerate(segments):
        seg["segment_id"] = idx

    return segments


def _build_segment(segment_id: int, frames: list,
                   room_type_hint: str, motion_by_id: dict) -> dict:
    frame_ids = [f["frame_id"] for f in frames]
    n         = len(frames)

    # ── room_type: weighted confidence voting ─────────────────────────────────
    # Each frame contributes its full confidence to its room_type vote and a
    # fractional credit to alternative_type when confidence is low.  A single
    # uncertain frame cannot flip the segment label, but many low-confidence
    # frames agreeing on the same alternative can influence the result.
    room_type_votes: Counter = Counter()
    for f in frames:
        rt   = f.get("room_type", "unknown")
        conf = f.get("confidence", 0.5)
        alt  = f.get("alternative_type")

        if rt != "unknown":
            room_type_votes[rt] += conf

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

    # ── boundary heuristic: optical-flow rotation + VLM significant_change ────
    # Two independent signals: either alone is sufficient to mark a boundary.
    rotations         = [abs(motion_by_id.get(fid, {}).get("rotation_deg", 0.0))
                         for fid in frame_ids]
    high_rot_count    = sum(1 for r in rotations if r >= _ROTATION_BOUNDARY_DEG)
    rotation_boundary = (high_rot_count / n) >= _BOUNDARY_FRAC

    sig_change_count  = sum(
        1 for f in frames
        if f.get("changes_from_previous", {}).get("significant_change", False)
    )
    vlm_boundary      = (sig_change_count / n) >= _VLM_BOUNDARY_FRAC

    is_boundary = rotation_boundary or vlm_boundary

    # ── motion aggregation: Stage 1 optical flow + Stage 2 VLM movement ───────
    motions       = [motion_by_id.get(fid, {}) for fid in frame_ids]
    directions    = [
        m.get("motion_direction", "unknown")
        for m in motions
        if m.get("motion_direction") not in (None, "unknown")
    ]
    all_rotations = [m.get("rotation_deg", 0.0) for m in motions]

    vlm_movements = [
        f.get("changes_from_previous", {}).get("camera_movement", "unknown")
        for f in frames
        if f.get("changes_from_previous", {}).get("camera_movement", "unknown") != "unknown"
    ]

    seg_direction = Counter(directions).most_common(1)[0][0] if directions else "forward"
    vlm_direction = Counter(vlm_movements).most_common(1)[0][0] if vlm_movements else "unknown"
    seg_rotation  = round(sum(all_rotations) / len(all_rotations), 1) if all_rotations else 0.0

    # ── spatial characteristics: top-5 by cross-frame frequency ──────────────
    char_counter: Counter = Counter()
    for f in frames:
        for c in f.get("spatial_characteristics", []):
            char_counter[c] += 1
    spatial_chars = [c for c, _ in char_counter.most_common(5)]

    # ── objects visible: top-10 by cross-frame frequency ─────────────────────
    obj_counter: Counter = Counter()
    for f in frames:
        for o in f.get("objects_visible", []):
            obj_counter[o] += 1
    objects_visible = [o for o, _ in obj_counter.most_common(10)]

    # ── view descriptions: unique per-frame summaries (up to 5) ──────────────
    seen_vd: set = set()
    view_descriptions: list = []
    for f in frames:
        vd = f.get("view_description", "")
        if vd and vd != "unknown" and vd not in seen_vd:
            seen_vd.add(vd)
            view_descriptions.append(vd)
            if len(view_descriptions) == 5:
                break

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
        "objects_visible":         objects_visible,
        "view_descriptions":       view_descriptions,
        "segment_motion": {
            "direction":     seg_direction,
            "vlm_direction": vlm_direction,
            "rotation_deg":  seg_rotation,
        },
        "frame_ids":   frame_ids,
        "confidence":  avg_confidence,
    }
