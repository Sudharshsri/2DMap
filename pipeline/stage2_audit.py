"""
Stage 2 — Self-consistency audit.

Scans the VLM perception list for internal contradictions before Stage 3
commits to segments. No extra VLM calls — pure logic over the JSON that
Stage 2 already produced.

Seven checks
------------
1. lone_outlier        A single frame disagrees with both neighbours (conf < 0.70).
2. contradictory_sig   significant_change=True but both VLM camera_movement and
                       Stage-1 optical-flow rotation say the camera was static.
3. uncertain_no_alt    Confidence < 0.50 with no alternative_type offered.
4. flip_no_transition  Room type changed with no rotation or sig-change evidence.
5. confidence_drop     Confidence fell > 0.30 in one frame (mid-doorway suspicion).
6. ghost_door          A door side reported in only 1 frame out of a 5-frame window
                       is flagged as a hallucination and removed from that frame.
7. open_door_entry     If camera moved "forward" and a door was open in the previous
                       frame, that door is tagged likely_traversed=True so Stage 4
                       knows which door the camera most likely passed through.

Auto-corrections (applied before Stage 3 sees the data)
---------------------------------------------------------
  lone_outlier       → room_type overridden to the surrounding majority type.
  contradictory_sig  → significant_change downgraded to False when both VLM and
                       optical-flow independently confirm the camera was static.
  ghost_door         → door entry removed from doors_visible when it does not
                       appear in ≥ 2 out of 5 surrounding frames.
  open_door_entry    → likely_traversed=True added to the open door in the prior
                       frame when the camera subsequently moved forward.

All corrections and every warning are recorded in the returned dict and saved
to output/stage2_audit.json for human review.
"""
from __future__ import annotations
from collections import Counter


def audit_perceptions(perceptions: list, frame_motion: list) -> dict:
    """
    Check Stage 2 perception list for self-consistency issues.

    Parameters
    ----------
    perceptions  : list of perception dicts (Stage 2 output)
    frame_motion : list of motion dicts    (Stage 1 output)

    Returns
    -------
    {
        "warnings":              [...],
        "corrected_perceptions": [...],   # feed this into Stage 3
        "summary":               {...},
    }
    """
    n          = len(perceptions)
    motion_map = {m["frame_id"]: m for m in frame_motion}
    warnings: list[dict] = []
    corrected = [dict(p) for p in perceptions]

    for i, fp in enumerate(perceptions):
        fid  = fp["frame_id"]
        rt   = fp["room_type"]
        conf = fp.get("confidence", 0.5)
        ch   = fp.get("changes_from_previous", {})
        sig  = ch.get("significant_change", False)
        move = ch.get("camera_movement", "unknown")
        rot  = abs(motion_map.get(fid, {}).get("rotation_deg", 0.0))

        # ── Check 1: lone outlier ────────────────────────────────────────────
        if 0 < i < n - 1:
            prev_rt = perceptions[i - 1]["room_type"]
            next_rt = perceptions[i + 1]["room_type"]
            if (rt not in ("unknown",) and rt != prev_rt and rt != next_rt
                    and prev_rt == next_rt and conf < 0.70):
                warnings.append(_warn(fid, "lone_outlier", "high",
                    f"room_type='{rt}' (conf={conf:.2f}) is isolated between "
                    f"two '{prev_rt}' frames — corrected to '{prev_rt}'"))
                corrected[i] = {**corrected[i],
                                "room_type":        prev_rt,
                                "_audit_corrected": "lone_outlier"}

        # ── Check 2: contradictory significant_change ────────────────────────
        if sig and move == "static" and rot < 15.0:
            warnings.append(_warn(fid, "contradictory_sig", "medium",
                f"significant_change=True but camera_movement=static and "
                f"optical rotation={rot:.1f}° — downgraded to False"))
            new_ch = {**ch, "significant_change": False}
            corrected[i] = {**corrected[i],
                            "changes_from_previous": new_ch,
                            "_audit_corrected":      "contradictory_sig"}

        # ── Check 3: uncertain with no fallback ──────────────────────────────
        if conf < 0.50 and fp.get("alternative_type") is None:
            warnings.append(_warn(fid, "uncertain_no_alt", "medium",
                f"confidence={conf:.2f} but no alternative_type — "
                f"room_type='{rt}' is a low-confidence guess with no fallback"))

        # ── Check 4: room type flip without any transition evidence ──────────
        if i > 0:
            prev_rt = perceptions[i - 1]["room_type"]
            if (rt not in ("unknown",) and prev_rt not in ("unknown",)
                    and rt != prev_rt
                    and not sig
                    and move not in ("rotating_left", "rotating_right")
                    and rot < 15.0):
                warnings.append(_warn(fid, "flip_no_transition", "high",
                    f"room_type changed '{prev_rt}'→'{rt}' but sig=False, "
                    f"move={move}, optical_rot={rot:.1f}° — "
                    f"may be real or a misclassification"))

        # ── Check 5: sudden confidence drop ──────────────────────────────────
        if i > 0:
            prev_conf = perceptions[i - 1].get("confidence", 0.5)
            if conf < prev_conf - 0.30:
                warnings.append(_warn(fid, "confidence_drop", "low",
                    f"confidence dropped {prev_conf:.2f}→{conf:.2f} "
                    f"— frame may be mid-doorway or motion-blurred"))

    # ── Check 6: ghost door (per-door hallucination in 5-frame window) ───────
    corrected = _check_ghost_doors(corrected, warnings)

    # ── Check 7: open-door / camera-entry cross-check ────────────────────────
    corrected = _check_open_door_entry(corrected, warnings)

    high   = sum(1 for w in warnings if w["severity"] == "high")
    medium = sum(1 for w in warnings if w["severity"] == "medium")
    low    = sum(1 for w in warnings if w["severity"] == "low")
    fixes  = sum(1 for p in corrected if "_audit_corrected" in p)

    summary = {
        "total_frames":        n,
        "warnings_count":      len(warnings),
        "high":                high,
        "medium":              medium,
        "low":                 low,
        "corrections_applied": fixes,
    }

    _print_report(warnings, summary)

    return {
        "warnings":              warnings,
        "corrected_perceptions": corrected,
        "summary":               summary,
    }


# ── Check 6: ghost door removal ──────────────────────────────────────────────

def _check_ghost_doors(corrected: list, warnings: list) -> list:
    """
    For each frame i, look at a 5-frame window centred on i.
    If a door side appears in frame i but in fewer than 2 of the 5 surrounding
    frames, it is a likely hallucination and is removed from frame i's
    doors_visible list.

    The 'none' pseudo-door is never removed (it's the absence-of-door marker).
    """
    n      = len(corrected)
    half   = 2   # window half-width → 5-frame window: [i-2 .. i+2]
    result = [dict(p) for p in corrected]

    for i, fp in enumerate(corrected):
        raw_doors = fp.get("doors_visible", [])
        real_doors = [d for d in raw_doors if d.get("side") != "none"]
        if not real_doors:
            continue

        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        # Count how many frames in the window also report each door side
        side_counts: Counter = Counter()
        for j in range(lo, hi):
            if j == i:
                continue
            for d in corrected[j].get("doors_visible", []):
                s = d.get("side", "none")
                if s != "none":
                    side_counts[s] += 1

        # Remove doors that do not appear in at least 1 OTHER frame in the window
        # (total window is 4 other frames; require ≥ 1 confirmation)
        kept   = []
        removed = []
        for d in raw_doors:
            side = d.get("side", "none")
            if side == "none":
                kept.append(d)
            elif side_counts.get(side, 0) >= 1:
                kept.append(d)
            else:
                removed.append(side)

        if removed:
            warnings.append(_warn(
                fp["frame_id"], "ghost_door", "medium",
                f"Door(s) {removed} appear in only this frame within a "
                f"5-frame window — removed as likely hallucination"
            ))
            if not kept:
                kept = [{"side": "none", "open": False,
                         "leads_to": "unknown", "confidence": 0.0}]
            result[i] = {**result[i],
                         "doors_visible":    kept,
                         "_audit_corrected": result[i].get("_audit_corrected", "") + "|ghost_door"}

    return result


# ── Check 7: open-door entry cross-check ─────────────────────────────────────

def _check_open_door_entry(corrected: list, warnings: list) -> list:
    """
    If frame i shows camera_movement=forward AND frame i-1 had at least one
    open door, tag that open door as likely_traversed=True.

    This gives Stage 4 a signal for which door the camera physically passed
    through, even when the transition is ambiguous.
    """
    result = [dict(p) for p in corrected]

    for i in range(1, len(corrected)):
        curr_ch   = corrected[i].get("changes_from_previous", {})
        curr_move = curr_ch.get("camera_movement", "unknown")

        if curr_move != "forward":
            continue

        prev = corrected[i - 1]
        prev_doors = prev.get("doors_visible", [])
        open_doors = [d for d in prev_doors if d.get("open", False) and d.get("side") != "none"]

        if not open_doors:
            continue

        # Tag the highest-confidence open door as likely_traversed
        best = max(open_doors, key=lambda d: d.get("confidence", 0.0))
        warnings.append(_warn(
            prev["frame_id"], "open_door_entry", "low",
            f"Camera moved forward in next frame; open door on '{best['side']}' "
            f"in this frame tagged likely_traversed=True"
        ))

        # Update the previous frame's door list in result
        new_doors = []
        tagged    = False
        for d in prev_doors:
            if not tagged and d.get("side") == best["side"] and d.get("open", False):
                new_doors.append({**d, "likely_traversed": True})
                tagged = True
            else:
                new_doors.append(d)

        result[i - 1] = {**result[i - 1], "doors_visible": new_doors}

    return result


# ── private helpers ───────────────────────────────────────────────────────────

def _warn(frame_id: int, check: str, severity: str, message: str) -> dict:
    return {"frame_id": frame_id, "check": check,
            "severity": severity, "message": message}


def _print_report(warnings: list, summary: dict) -> None:
    label = {"high": "[HIGH] ", "medium": "[WARN] ", "low": "[INFO] "}
    for w in warnings:
        print(f"  {label.get(w['severity'], '')}"
              f"frame {w['frame_id']:>2}  {w['check']}:  {w['message']}")
    n = summary["warnings_count"]
    c = summary["corrections_applied"]
    h = summary["high"]
    if n == 0:
        print("  All frames passed all consistency checks.")
    else:
        print(f"\n  Summary : {n} warnings  ({h} high)  |  "
              f"{c} auto-correction(s) applied")
