"""
Stage 1 — Video preprocessing.

Extracts frames at 1 fps and computes per-frame motion heuristics
using frame differencing + Lucas-Kanade optical flow (CPU-only).
"""
import os
import cv2
import numpy as np
from pathlib import Path

# Mean-pixel-diff below this → camera is considered static
_STATIC_THRESHOLD = 5.0

# Lucas-Kanade parameters
_FEATURE_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def extract_frames_with_motion(video_path: str, output_dir: str,
                                fps_target: float = 1.0) -> dict:
    """
    Extract frames from *video_path* at *fps_target* fps and compute
    per-frame motion heuristics.

    Returns
    -------
    {
        "frame_paths": [...],   # absolute paths to saved JPEG frames
        "frame_motion": [...]   # list of motion dicts (see _compute_motion)
    }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_video   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_raw   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval    = max(1, int(round(fps_video / fps_target)))

    print(f"  Video  : {fps_video:.1f} fps  |  {total_raw} raw frames")
    print(f"  Saving : 1 frame every {interval} raw frames  (~{fps_target} fps)")

    frame_paths: list[str] = []
    frame_motion: list[dict] = []
    prev_gray = None
    saved_idx = 0
    raw_idx   = 0

    accum_dx = 0.0
    accum_dy = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow_dx, flow_dy = _compute_raw_flow(prev_gray, gray)
            accum_dx += flow_dx
            accum_dy += flow_dy

        if raw_idx % interval == 0:
            fname = f"frame_{saved_idx:04d}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame)
            frame_paths.append(fpath)

            direction, rotation = _interpret_flow(accum_dx, accum_dy)
            frame_motion.append({
                "frame_id": saved_idx,
                "motion_direction": direction,
                "rotation_deg": round(rotation, 1)
            })

            # Reset accumulators for the next interval
            accum_dx = 0.0
            accum_dy = 0.0
            saved_idx += 1

            if saved_idx % 20 == 0:
                print(f"  ... {saved_idx} frames saved")

        prev_gray = gray
        raw_idx += 1

    cap.release()
    print(f"  Done   : {saved_idx} frames extracted")
    return {"frame_paths": frame_paths, "frame_motion": frame_motion}


# ── private helpers ──────────────────────────────────────────────────────────

def _compute_raw_flow(prev_gray, curr_gray) -> tuple[float, float]:
    """Return raw (dx, dy) optical flow displacement between consecutive frames."""
    diff      = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = float(np.mean(diff))

    if mean_diff < _STATIC_THRESHOLD:
        return 0.0, 0.0

    # Lucas-Kanade sparse optical flow
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **_FEATURE_PARAMS)
    if p0 is None or len(p0) < 5:
        return 0.0, 0.0

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None,
                                          **_LK_PARAMS)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    if len(good_new) < 3:
        return 0.0, 0.0

    flow = good_new - good_old
    dx   = float(np.mean(flow[:, 0]))
    dy   = float(np.mean(flow[:, 1]))

    return dx, dy


def _interpret_flow(dx: float, dy: float) -> tuple:
    """Map average optical-flow displacement to (motion_direction, rotation_deg)."""
    adx, ady = abs(dx), abs(dy)

    if adx < 2.0 and ady < 2.0:
        return "unknown", 0.0

    if adx > ady * 2.0:              # predominantly lateral
        if dx < 0:
            return ("right",  90.0) if adx > 20 else ("right",  45.0)
        else:
            return ("left",  -90.0) if adx > 20 else ("left",  -45.0)

    if ady > adx * 2.0:              # predominantly vertical
        return ("forward", 0.0) if dy > 0 else ("backward", 180.0)

    # diagonal
    return ("right", 30.0) if dx < 0 else ("left", -30.0)
