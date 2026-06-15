"""
Stage 2 — Per-frame semantic perception using Qwen2.5-VL-3B-Instruct
loaded locally via HuggingFace transformers.

Changes from v1
---------------
* Removed 'hallway' from VALID_ROOM_TYPES — it is collapsed into 'corridor'
  via SEMANTIC_CANONICAL and an explicit prompt guideline, eliminating the
  corridor/hallway synonym collision that caused single-frame outlier segments.
* Removed 'is_boundary_heuristic' from the VLM prompt and output dict.
  Boundary detection is now done in Stage 3 using motion rotation data,
  which is a more reliable signal than asking a VLM to predict pipeline state.
* Added 'alternative_type' field.  When the model's confidence is below 0.65
  it records its second-best guess; Stage 3 uses this for weighted voting
  instead of a hard room-type flip.
* Prompt reframed as observational rather than purely categorical — the model
  is asked to describe what it sees and then map, rather than just pick a label.
* view_direction renamed to view_description for clarity.

RAM note (CPU-only, 16 GB):
  float16 weights alone take ~6 GB for 3B, leaving comfortable headroom.
  If you need even lighter, there is no official Qwen2.5-VL 1B yet; fall
  back to the default perception if the model fails to load.

Requirements:
    transformers>=4.49  qwen-vl-utils  torch  torchvision  accelerate  Pillow
"""
import json
import re
import torch
from pathlib import Path
from PIL import Image

# ── model selection ───────────────────────────────────────────────────────────

_HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_MIN_PIXELS  = 128 * 28 * 28
_MAX_PIXELS  = 256 * 28 * 28

# ── semantic normalization ────────────────────────────────────────────────────
# Applied at parse time to collapse VLM synonyms that share the same physical
# space.  This is the primary defence against label collisions; Stage 3 applies
# the same map as a secondary safety net.

SEMANTIC_CANONICAL: dict[str, str] = {
    "hallway":    "corridor",
    "passage":    "corridor",
    "walkway":    "corridor",
    "passageway": "corridor",
}

# ── allowed vocabulary ────────────────────────────────────────────────────────
# 'hallway' is intentionally absent — it is handled by SEMANTIC_CANONICAL above
# and excluded from the prompt so the model never samples it as a primary label.

VALID_ROOM_TYPES = {
    "entrance", "corridor", "living_room", "bedroom",
    "kitchen", "bathroom", "office", "stairwell", "lobby", "unknown",
}
VALID_SIZE_HINTS = {"very_small", "small", "medium", "large", "very_large"}
VALID_DOOR_SIDES = {"left", "right", "front", "none"}

# ── prompt ────────────────────────────────────────────────────────────────────

_PROMPT = """\
You are analysing a single frame from an indoor walkthrough video.
Describe ONLY what is clearly and directly visible in the image.

Output ONLY this JSON (no markdown, no explanation):
{
  "room_type": "<entrance|corridor|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown>",
  "alternative_type": "<second-best guess from the same list, or null if confident>",
  "size_hint": "<very_small|small|medium|large|very_large>",
  "doors_visible": ["<left|right|front|none>"],
  "view_description": "<3-6 words describing what is literally visible>",
  "spatial_characteristics": ["<observed feature 1>", "<observed feature 2>"],
  "confidence": <0.1-1.0>
}

Guidelines:
- If the space is a long narrow passage, walkway, or hallway of any kind, always output "corridor".
- If you cannot clearly identify the space type, output "unknown" with confidence below 0.4.
- Do NOT invent rooms, doors, or objects that are not clearly visible.
- spatial_characteristics must describe physically observable things
  (e.g. "tiled floor", "glass door on the left", "exposed brick wall").
- Set confidence below 0.6 if the view is partial, blurry, or transitional.
- If confidence is below 0.65, provide a meaningful alternative_type; otherwise set it to null."""


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
        alt_str = f" (alt={perc['alternative_type']})" if perc.get("alternative_type") else ""
        print(
            f"  -> {perc['room_type']:<20}{alt_str}"
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
                    {"type": "image", "image": image},
                    {"type": "text",  "text": _PROMPT},
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

def _normalize_room_type(rt: str) -> str:
    """Canonicalize a raw room-type string: lowercase, apply synonym map, validate."""
    rt = rt.strip().lower().replace(" ", "_")
    rt = SEMANTIC_CANONICAL.get(rt, rt)
    return rt if rt in VALID_ROOM_TYPES else "unknown"


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

    # --- room_type (normalised) ---
    room_type = _normalize_room_type(str(raw.get("room_type", "unknown")))

    # --- alternative_type (new) ---
    # Only kept when it differs from room_type and is not unknown.
    alt_raw = raw.get("alternative_type")
    alternative_type: str | None = None
    if alt_raw and str(alt_raw).strip().lower() not in ("null", "none", ""):
        alt_candidate = _normalize_room_type(str(alt_raw))
        if alt_candidate not in ("unknown", room_type):
            alternative_type = alt_candidate

    # --- size_hint ---
    sh = str(raw.get("size_hint", "medium")).lower().replace(" ", "_")
    size_hint = sh if sh in VALID_SIZE_HINTS else "medium"

    # --- doors_visible ---
    raw_doors = raw.get("doors_visible", ["none"])
    if not isinstance(raw_doors, list):
        raw_doors = [str(raw_doors)]
    doors_visible = [
        d.lower() for d in raw_doors if str(d).lower() in VALID_DOOR_SIDES
    ] or ["none"]

    # --- view_description (renamed from view_direction) ---
    # Accept either key for backwards-compat with cached stage2 results.
    vd = str(raw.get("view_description", raw.get("view_direction", "unknown"))).strip().lower()
    view_description = re.sub(r"[^\w\s]", "", vd)[:80] or "unknown"

    # --- spatial_characteristics ---
    sc = raw.get("spatial_characteristics", [])
    spatial = [str(c).strip() for c in sc if str(c).strip()][:5] \
        if isinstance(sc, list) else []

    # --- confidence ---
    try:
        conf = round(max(0.1, min(1.0, float(raw.get("confidence", 0.5)))), 2)
    except (TypeError, ValueError):
        conf = 0.5

    # Guard: if the model claims high confidence but still offered an alternative,
    # cap conf at 0.70 so Stage 3 weighted voting gets a fair signal.
    if alternative_type and conf > 0.70:
        conf = 0.70

    return {
        "frame_id":                frame_id,
        "room_type":               room_type,
        "alternative_type":        alternative_type,
        "size_hint":               size_hint,
        "doors_visible":           doors_visible,
        "view_description":        view_description,
        "spatial_characteristics": spatial,
        "confidence":              conf,
    }


def _default_perception(frame_id: int) -> dict:
    return {
        "frame_id":                frame_id,
        "room_type":               "unknown",
        "alternative_type":        None,
        "size_hint":               "medium",
        "doors_visible":           ["none"],
        "view_description":        "unknown",
        "spatial_characteristics": [],
        "confidence":              0.1,
    }