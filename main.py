import os
import json
from step0_parse_is_962    import extract_is962_context
from step1_extract_frames import extract_key_frames
from step2_vlm_process    import process_frames_with_vlm
from step3_generate_dxf   import generate_floor_plan

def run():
    print("=" * 55)
    print("  Floor Plan Generator — IS 962 Compliant (Moondream)")
    print("=" * 55)

    os.makedirs("output", exist_ok=True)

    # Step 1: Read IS 962 PDF → extract context
    print("\n[1/4] Parsing IS 962 PDF...")
    is962_ctx = extract_is962_context("input/IS 962.pdf")
    print(f"  → IS 962 context ready ({len(is962_ctx)} characters)")

    # Step 2: Extract frames from video (1 fps)
    print("\n[2/4] Extracting video frames...")
    frames = extract_key_frames("input/room_video.mp4", "output/frames", frames_per_second=1)
    print(f"  → {len(frames)} frames extracted")

    # Step 3: Moondream analyzes frames using IS 962 context → JSON
    print("\n[3/4] Running Moondream VLM analysis...")
    data = process_frames_with_vlm("output/frames", is962_ctx)
    with open("output/floor_plan.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Rooms: {len(data.get('rooms', []))}  "
          f"Doors: {len(data.get('doors', []))}  "
          f"Windows: {len(data.get('windows', []))}")

    # Step 4: Build DXF + PNG from JSON
    print("\n[4/4] Generating DXF floor plan...")
    generate_floor_plan("output/floor_plan.json",
                        "output/floor_plan.dxf",
                        "output/floor_plan.png")

    print("\n✅ Pipeline complete!")
    print("   → output/floor_plan.dxf")
    print("   → output/floor_plan.png")
    print("   → output/floor_plan.json")

if __name__ == "__main__":
    run()