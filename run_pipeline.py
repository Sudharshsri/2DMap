#!/usr/bin/env python3
"""
2DMap — end-to-end indoor floor-plan generator.

Usage
-----
  python run_pipeline.py --video input/room_video.mp4

Caching
-------
Each stage writes its result to output/stageN_*.json.
Re-run with --skip 1 2 (etc.) to skip already-completed stages
and load their cached JSON instead.

Stages
------
  1  Extract frames + motion heuristics  (OpenCV)
  2  Per-frame semantic perception        (Moondream VLM)
  3  Segment grouping + transition detect (pure Python)
  4  Global floor-plan structuring        (Llama 3.2:3b via Ollama)
  5  CAD rendering                        (ezdxf + Matplotlib)
"""
import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows so Unicode characters print without crashing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate a 2D floor plan from an indoor walkthrough video."
    )
    p.add_argument(
        "--video", default="input/room_video.mp4",
        help="Path to the input video file (default: input/room_video.mp4)",
    )
    p.add_argument(
        "--output", default="output",
        help="Output directory (default: output/)",
    )
    p.add_argument(
        "--fps", type=int, default=1,
        help="Frame extraction rate in fps (default: 1)",
    )
    p.add_argument(
        "--skip", type=int, nargs="*", default=[],
        metavar="N",
        help="Stage numbers to skip by loading cached JSON (e.g. --skip 1 2)",
    )
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _header(n: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Stage {n}  --  {title}")
    print(f"{'='*60}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    out_dir    = Path(args.output)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    skip = set(args.skip)

    # ── Stage 1: Frame extraction + motion ───────────────────────────────────
    _header(1, "Frame extraction & motion heuristics")
    cache1 = out_dir / "stage1_result.json"

    if 1 in skip and cache1.exists():
        print("  Skipping — loading cache.")
        s1 = _load_json(cache1)
    else:
        from pipeline.stage1_preprocess import extract_frames_with_motion
        s1 = extract_frames_with_motion(str(video_path), str(frames_dir), args.fps)
        _save_json(cache1, s1)

    frame_paths  = s1["frame_paths"]
    frame_motion = s1["frame_motion"]
    print(f"  Result : {len(frame_paths)} frames, {len(frame_motion)} motion entries")

    # ── Stage 2: Moondream perception ─────────────────────────────────────────
    _header(2, "Per-frame semantic perception (Moondream)")
    cache2 = out_dir / "stage2_perception.json"

    if 2 in skip and cache2.exists():
        print("  Skipping — loading cache.")
        frame_perception = _load_json(cache2)
    else:
        from pipeline.stage2_perception import analyze_frames
        frame_perception = analyze_frames(frame_paths)
        _save_json(cache2, frame_perception)

    print(f"  Result : {len(frame_perception)} frame perceptions")

    # ── Stage 3: Segmentation + transitions ───────────────────────────────────
    _header(3, "Segment grouping & transition detection")
    cache3 = out_dir / "stage3_segments.json"

    if 3 in skip and cache3.exists():
        print("  Skipping — loading cache.")
        s3          = _load_json(cache3)
        segments    = s3["segments"]
        transitions = s3["transitions"]
    else:
        from pipeline.stage3_segmentation import segment_frames, detect_transitions
        segments    = segment_frames(frame_perception, frame_motion)
        transitions = detect_transitions(segments)
        _save_json(cache3, {"segments": segments, "transitions": transitions})

    print(f"  Result : {len(segments)} segments, {len(transitions)} transitions")
    for seg in segments:
        print(f"    Seg {seg['segment_id']:>2}  {seg['room_type']:<20}"
              f"  frames={len(seg['frame_ids'])}"
              f"  size={seg['size_hint']}"
              f"  doors={[d['side'] for d in seg['door_locations']]}")

    # ── Stage 4: LLM floor-plan structuring ───────────────────────────────────
    _header(4, "Floor-plan structuring (Llama 3.2:3b via Ollama)")
    cache4 = out_dir / "stage4_floor_plan.json"

    if 4 in skip and cache4.exists():
        print("  Skipping — loading cache.")
        floor_plan = _load_json(cache4)
    else:
        from pipeline.stage4_llm_structuring import generate_floor_plan
        floor_plan = generate_floor_plan(segments, transitions)
        _save_json(cache4, floor_plan)

    rooms = floor_plan.get("rooms", [])
    print(f"  Result : {len(rooms)} rooms, "
          f"{len(floor_plan.get('transitions', []))} transitions, "
          f"{len(floor_plan.get('camera_path', []))} camera waypoints")
    for r in rooms:
        print(f"    {r['id']}  {r['type']:<20}  "
              f"{r['width']}m × {r['height']}m  ({r['size_hint']})")

    # ── Stage 5: CAD rendering ────────────────────────────────────────────────
    _header(5, "CAD rendering  →  DXF + PNG")
    dxf_path = str(out_dir / "floor_plan.dxf")
    png_path = str(out_dir / "floor_plan.png")

    from pipeline.stage5_cad_rendering import render_floor_plan
    render_floor_plan(floor_plan, dxf_path, png_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")
    print(f"  DXF   : {dxf_path}")
    print(f"  PNG   : {png_path}")
    print(f"  Cache : {out_dir}/stage*.json")
    print()


if __name__ == "__main__":
    main()
