"""
Stage 2 — Per-frame semantic perception using Qwen2.5-VL-7B-Instruct
loaded locally via HuggingFace transformers.

First run downloads ~15 GB of weights to the HuggingFace cache.
Subsequent runs load from cache instantly.

RAM note (CPU-only, 16 GB):
  float16 weights alone take ~14 GB, leaving ~2 GB for activations.
  If you hit an OOM error, change _HF_MODEL_ID to the 3B variant:
      "Qwen/Qwen2.5-VL-3B-Instruct"   (~6 GB in float16, very comfortable)

Requirements (already installed):
    transformers>=4.49  qwen-vl-utils  torch  torchvision  accelerate  Pillow
"""
import json
import re
import torch
from pathlib import Path
from PIL import Image

# ── model selection ───────────────────────────────────────────────────────────

_HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Reduce image resolution to save activation memory on CPU (default is 1280*28*28)
_MIN_PIXELS = 128 * 28 * 28
_MAX_PIXELS  = 256 * 28 * 28

# ── allowed vocabulary ────────────────────────────────────────────────────────

VALID_SIZE_HINTS = {"very_small", "small", "medium", "large", "very_large"}
VALID_DOOR_SIDES = {"left", "right", "front", "none"}
VALID_ROOM_TYPES = {
    "entrance", "corridor", "hallway", "living_room", "bedroom",
    "kitchen", "bathroom", "office", "stairwell", "lobby", "unknown",
}

# ── prompt ────────────────────────────────────────────────────────────────────

_PROMPT = """\
You are analysing a single frame from an indoor walkthrough video.
Describe ONLY what is clearly visible. Do NOT invent rooms, doors, or objects.

Output ONLY this JSON (no markdown, no explanation):
{
  "room_type": "<entrance|corridor|hallway|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown>",
  "size_hint": "<very_small|small|medium|large|very_large>",
  "doors_visible": ["<left|right|front|none>"],
  "view_direction": "<3-6 word description>",
  "is_boundary_heuristic": <true|false>,
  "spatial_characteristics": ["<visible feature 1>", "<visible feature 2>"],
  "confidence": <0.1-1.0>
}"""


# ── public API ────────────────────────────────────────────────────────────────

def analyze_frames(frame_paths: list) -> list:
    """
    Run Qwen2.5-VL locally on every extracted frame.
    Returns a list of normalised perception dicts.
    """
    model, processor = _load_model()
    if model is None:
        print("  WARNING: Could not load Qwen2.5-VL — using default perceptions.")
        return [_default_perception(i) for i in range(len(frame_paths))]

    perceptions: list = []
    n = len(frame_paths)
    for i, fpath in enumerate(frame_paths):
        print(f"  Frame {i+1:>4}/{n}: {Path(fpath).name}", end="", flush=True)
        perc = _analyse_one(model, processor, fpath, i)
        perceptions.append(perc)
        print(
            f"  -> {perc['room_type']:<20}"
            f"  size={perc['size_hint']:<12}"
            f"  doors={perc['doors_visible']}"
            f"  conf={perc['confidence']:.2f}"
        )

    return perceptions


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"  Loading {_HF_MODEL_ID} …")
        print("  (first run downloads weights, subsequent runs use cache)")

        processor = AutoProcessor.from_pretrained(
            _HF_MODEL_ID,
            min_pixels=_MIN_PIXELS,
            max_pixels=_MAX_PIXELS,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _HF_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        print("  Model ready.\n")
        return model, processor

    except Exception as exc:
        print(f"\n  ERROR loading model: {exc}")
        if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
            print("  OOM — try changing _HF_MODEL_ID to 'Qwen/Qwen2.5-VL-3B-Instruct' (~6 GB)")
        return None, None


# ── per-frame inference ───────────────────────────────────────────────────────

def _analyse_one(model, processor, frame_path: str, frame_id: int) -> dict:
    try:
        from qwen_vl_utils import process_vision_info

        image = Image.open(frame_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image",  "image": image},
                    {"type": "text",   "text": _PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return _parse_response(response, frame_id)

    except Exception as exc:
        print(f"\n  WARNING: frame {frame_id} failed ({exc})")
        return _default_perception(frame_id)


# ── JSON parsing & normalisation ──────────────────────────────────────────────

def _parse_response(text: str, frame_id: int) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()

    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return _default_perception(frame_id)

    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return _default_perception(frame_id)

    rt = str(raw.get("room_type", "unknown")).lower().replace(" ", "_")
    room_type = rt if rt in VALID_ROOM_TYPES else "unknown"

    sh = str(raw.get("size_hint", "medium")).lower().replace(" ", "_")
    size_hint = sh if sh in VALID_SIZE_HINTS else "medium"

    raw_doors = raw.get("doors_visible", ["none"])
    if not isinstance(raw_doors, list):
        raw_doors = [str(raw_doors)]
    doors_visible = [
        d.lower() for d in raw_doors if str(d).lower() in VALID_DOOR_SIDES
    ] or ["none"]

    vd = str(raw.get("view_direction", "unknown")).strip().lower()
    view_direction = re.sub(r"[^\w]", "_", vd)[:60] or "unknown"

    bh = raw.get("is_boundary_heuristic", False)
    is_boundary = bool(bh) if isinstance(bh, bool) else "true" in str(bh).lower()

    sc = raw.get("spatial_characteristics", [])
    spatial = [str(c).strip() for c in sc if str(c).strip()][:5] \
        if isinstance(sc, list) else []

    try:
        conf = round(max(0.1, min(1.0, float(raw.get("confidence", 0.5)))), 2)
    except (TypeError, ValueError):
        conf = 0.5

    return {
        "frame_id":                frame_id,
        "room_type":               room_type,
        "size_hint":               size_hint,
        "doors_visible":           doors_visible,
        "view_direction":          view_direction,
        "is_boundary_heuristic":   is_boundary,
        "spatial_characteristics": spatial,
        "confidence":              conf,
    }


def _default_perception(frame_id: int) -> dict:
    return {
        "frame_id":                frame_id,
        "room_type":               "unknown",
        "size_hint":               "medium",
        "doors_visible":           ["none"],
        "view_direction":          "unknown",
        "is_boundary_heuristic":   False,
        "spatial_characteristics": [],
        "confidence":              0.1,
    }
