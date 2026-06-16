"""
Stage 2 — Per-frame semantic perception using Qwen2.5-VL-3B via Ollama.

Qwen2.5-VL is a vision-language model that reliably follows structured JSON
prompts. Running it through Ollama (GGUF Q4, ~2.2 GB) is much lighter than
loading it via HuggingFace (~6 GB float16) and requires no PyTorch.

Install:  ollama pull qwen2.5vl:3b

Two prompts are used:
  _PROMPT_FIRST     — first frame only; no previous frame to compare.
  _PROMPT_WITH_PREV — all subsequent frames; previous and current frame are
                      stitched side-by-side so the model sees both and can
                      describe exactly what changed between them.

Each perception dict includes:
  objects_visible       — every distinct object the model identifies
  changes_from_previous — camera_movement, new/disappeared elements,
                          significant_change flag, and a change description.

Stage 3 uses significant_change as an additional boundary signal on top of
the optical-flow rotation heuristic from Stage 1.
"""
import base64
import io
import json
import re
import requests
from pathlib import Path
from PIL import Image

# ── Ollama config ─────────────────────────────────────────────────────────────

_OLLAMA_URL  = "http://localhost:11434/api/generate"
_OLLAMA_TAGS = "http://localhost:11434/api/tags"
_MODEL       = "qwen2.5vl:3b"
_TIMEOUT_SEC = 180
_MAX_IMG_DIM = 512

# ── semantic normalization ────────────────────────────────────────────────────

SEMANTIC_CANONICAL: dict[str, str] = {
    "hallway":     "corridor",
    "passage":     "corridor",
    "walkway":     "corridor",
    "passageway":  "corridor",
    "hall":        "corridor",
    "foyer":       "entrance",
    "entryway":    "entrance",
    "lounge":      "living_room",
    "dining_room": "living_room",
    "restroom":    "bathroom",
    "toilet":      "bathroom",
    "washroom":    "bathroom",
    "study":       "office",
    "den":         "office",
}

VALID_ROOM_TYPES = {
    "entrance", "corridor", "living_room", "bedroom",
    "kitchen", "bathroom", "office", "stairwell", "lobby", "unknown",
}
VALID_SIZE_HINTS = {"very_small", "small", "medium", "large", "very_large"}
VALID_DOOR_SIDES = {"left", "right", "front", "none"}
VALID_MOVEMENTS  = {
    "forward", "backward", "left", "right",
    "rotating_left", "rotating_right", "static", "unknown",
}

# ── prompts ───────────────────────────────────────────────────────────────────

_PROMPT_FIRST = """\
You are analyzing the first frame of an indoor walkthrough video.
Describe in careful detail exactly what you see.

Output ONLY valid JSON — no markdown fences, no explanation:
{
  "room_type": "entrance|corridor|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown",
  "alternative_type": "second-best room type or null if confident",
  "size_hint": "very_small|small|medium|large|very_large",
  "doors_visible": ["left|right|front|none"],
  "view_description": "detailed 15-25 word description of what is literally visible",
  "objects_visible": ["every distinct object you can identify"],
  "spatial_characteristics": ["floor material", "wall material", "ceiling detail", "lighting"],
  "changes_from_previous": {
    "camera_movement": "unknown",
    "new_elements": [],
    "disappeared_elements": [],
    "significant_change": false,
    "change_description": "first frame, no previous to compare"
  },
  "confidence": 0.0
}

Rules:
- Any long narrow passage, hallway, or corridor: use "corridor".
- Cannot identify the space: use "unknown" with confidence below 0.4.
- List every visible object (furniture, fixtures, doors, windows, appliances).
- spatial_characteristics: physically observable details only.
- confidence: 0.8+ if very clear, 0.5-0.7 if partial view, below 0.5 if unclear.
- If confidence < 0.65, set a meaningful alternative_type; otherwise null."""

_PROMPT_WITH_PREV = """\
This image shows TWO consecutive indoor video frames placed side by side.
LEFT HALF = PREVIOUS frame.  RIGHT HALF = CURRENT frame.

TASK 1: Analyze the CURRENT frame (right half) in full detail.
TASK 2: Compare CURRENT to PREVIOUS and describe exactly what changed.

Output ONLY valid JSON — no markdown fences, no explanation:
{
  "room_type": "entrance|corridor|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown",
  "alternative_type": "second-best room type or null if confident",
  "size_hint": "very_small|small|medium|large|very_large",
  "doors_visible": ["left|right|front|none"],
  "view_description": "detailed 15-25 word description of the CURRENT frame (right half)",
  "objects_visible": ["every distinct object visible in the CURRENT frame"],
  "spatial_characteristics": ["floor material", "wall material", "ceiling detail", "lighting"],
  "changes_from_previous": {
    "camera_movement": "forward|backward|left|right|rotating_left|rotating_right|static",
    "new_elements": ["things that appeared in CURRENT not visible in PREVIOUS"],
    "disappeared_elements": ["things in PREVIOUS that are gone in CURRENT"],
    "significant_change": false,
    "change_description": "1-2 sentences on the most important change between frames"
  },
  "confidence": 0.0
}

Rules:
- room_type, view_description, objects_visible, spatial_characteristics: CURRENT frame only.
- changes_from_previous: compare CURRENT (right) to PREVIOUS (left).
- Any long narrow passage, hallway, or corridor: use "corridor".
- significant_change = true if a new room became visible, a door appeared or
  disappeared, or a dramatic viewpoint shift occurred; otherwise false.
- List every visible object in the CURRENT frame.
- confidence: 0.8+ if very clear, 0.5-0.7 if partial view, below 0.5 if unclear.
- If confidence < 0.65, set a meaningful alternative_type; otherwise null."""


# ── public API ────────────────────────────────────────────────────────────────

def analyze_frames(frame_paths: list) -> list:
    """
    Run Qwen2.5-VL via Ollama on every frame with temporal comparison.
    Returns a list of normalised perception dicts.
    """
    if not _check_ollama():
        print("  WARNING: Ollama not reachable at localhost:11434 — using default perceptions.")
        print("  Start Ollama and run:  ollama pull qwen2.5vl:3b")
        return [_default_perception(i) for i in range(len(frame_paths))]

    perceptions: list = []
    n = len(frame_paths)
    prev_image: Image.Image | None = None

    for i, fpath in enumerate(frame_paths):
        print(f"  Frame {i+1:>4}/{n}: {Path(fpath).name}", end="", flush=True)
        perc = _analyse_one(fpath, i, prev_image)
        perceptions.append(perc)

        try:
            prev_image = Image.open(fpath).convert("RGB")
        except Exception:
            prev_image = None

        delta   = perc.get("changes_from_previous", {})
        alt_str = f" (alt={perc['alternative_type']})" if perc.get("alternative_type") else ""
        print(
            f"  -> {perc['room_type']:<20}{alt_str}"
            f"  size={perc['size_hint']:<12}"
            f"  move={delta.get('camera_movement', '?'):<16}"
            f"  sig={str(delta.get('significant_change', False)):<6}"
            f"  conf={perc['confidence']:.2f}"
        )

    return perceptions


# ── Ollama connectivity ───────────────────────────────────────────────────────

def _check_ollama() -> bool:
    try:
        requests.get(_OLLAMA_TAGS, timeout=5)
        return True
    except Exception:
        return False


# ── image utilities ───────────────────────────────────────────────────────────

def _image_to_b64(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((_MAX_IMG_DIM, _MAX_IMG_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_side_by_side(prev: Image.Image, curr: Image.Image) -> str:
    """
    Stitch prev (left) and curr (right) at equal height, return base64 JPEG.
    Each half is capped at _MAX_IMG_DIM × _MAX_IMG_DIM before compositing so
    the total image never exceeds 2 × _MAX_IMG_DIM wide, keeping token count
    bounded even for wide-format source frames.
    """
    target_h = _MAX_IMG_DIM

    def _resize_h(img: Image.Image, h: int) -> Image.Image:
        scale = h / img.height
        w = max(1, int(img.width * scale))
        # also cap width so a landscape frame doesn't make the composite too wide
        if w > _MAX_IMG_DIM:
            w = _MAX_IMG_DIM
            scale = w / img.width
            h = max(1, int(img.height * scale))
        return img.resize((w, h), Image.LANCZOS)

    left  = _resize_h(prev, target_h)
    right = _resize_h(curr, target_h)

    divider_w = 2
    composite = Image.new("RGB", (left.width + divider_w + right.width, target_h),
                          color=(200, 200, 200))
    composite.paste(left,  (0, 0))
    composite.paste(right, (left.width + divider_w, 0))

    buf = io.BytesIO()
    composite.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Ollama inference ──────────────────────────────────────────────────────────

def _call_ollama(prompt: str, b64: str, retries: int = 2) -> str | None:
    for attempt in range(retries):
        try:
            resp = requests.post(
                _OLLAMA_URL,
                json={
                    "model":   _MODEL,
                    "prompt":  prompt,
                    "images":  [b64],
                    "stream":  False,
                    "options": {"temperature": 0.1, "num_predict": 700},
                },
                timeout=_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            print(f"\n  Ollama connection error (attempt {attempt+1}/{retries})")
        except Exception as exc:
            print(f"\n  Ollama error: {exc} (attempt {attempt+1}/{retries})")
    return None


# ── per-frame inference ───────────────────────────────────────────────────────

def _analyse_one(frame_path: str, frame_id: int,
                 prev_image: Image.Image | None) -> dict:
    try:
        curr_image = Image.open(frame_path).convert("RGB")

        if prev_image is None:
            b64 = _image_to_b64(curr_image)
            raw = _call_ollama(_PROMPT_FIRST, b64)
        else:
            b64 = _make_side_by_side(prev_image, curr_image)
            raw = _call_ollama(_PROMPT_WITH_PREV, b64)

        if raw is None:
            return _default_perception(frame_id)

        return _parse_response(raw, frame_id)

    except Exception as exc:
        print(f"\n  WARNING: frame {frame_id} failed ({exc})")
        return _default_perception(frame_id)


# ── JSON parsing & normalisation ──────────────────────────────────────────────

def _normalize_room_type(rt: str) -> str:
    rt = rt.strip().lower().replace(" ", "_")
    rt = SEMANTIC_CANONICAL.get(rt, rt)
    return rt if rt in VALID_ROOM_TYPES else "unknown"


def _clean_str_list(lst, max_n: int) -> list:
    if not isinstance(lst, list):
        return []
    return [str(e).strip() for e in lst if str(e).strip()][:max_n]


def _parse_response(text: str, frame_id: int) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().replace("```", "")

    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return _default_perception(frame_id)

    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return _default_perception(frame_id)

    # --- room_type ---
    room_type = _normalize_room_type(str(raw.get("room_type", "unknown")))

    # --- alternative_type ---
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

    # --- view_description ---
    vd = str(raw.get("view_description", raw.get("view_direction", "unknown"))).strip().lower()
    view_description = re.sub(r"[^\w\s]", "", vd)[:120] or "unknown"

    # --- objects_visible ---
    objects_visible = _clean_str_list(raw.get("objects_visible", []), 10)

    # --- spatial_characteristics ---
    spatial = _clean_str_list(raw.get("spatial_characteristics", []), 5)

    # --- changes_from_previous ---
    raw_ch = raw.get("changes_from_previous", {})
    if not isinstance(raw_ch, dict):
        raw_ch = {}

    cam_move = str(raw_ch.get("camera_movement", "unknown")).lower().replace(" ", "_")
    if cam_move not in VALID_MOVEMENTS:
        cam_move = "unknown"

    sig_raw = raw_ch.get("significant_change", False)
    if isinstance(sig_raw, str):
        significant_change = sig_raw.strip().lower() in ("true", "1", "yes")
    else:
        significant_change = bool(sig_raw)

    changes_from_previous = {
        "camera_movement":      cam_move,
        "new_elements":         _clean_str_list(raw_ch.get("new_elements", []), 5),
        "disappeared_elements": _clean_str_list(raw_ch.get("disappeared_elements", []), 5),
        "significant_change":   significant_change,
        "change_description":   str(raw_ch.get("change_description", ""))[:200],
    }

    # --- confidence ---
    try:
        conf = round(max(0.1, min(1.0, float(raw.get("confidence", 0.5)))), 2)
    except (TypeError, ValueError):
        conf = 0.5

    if alternative_type and conf > 0.70:
        conf = 0.70

    return {
        "frame_id":                frame_id,
        "room_type":               room_type,
        "alternative_type":        alternative_type,
        "size_hint":               size_hint,
        "doors_visible":           doors_visible,
        "view_description":        view_description,
        "objects_visible":         objects_visible,
        "spatial_characteristics": spatial,
        "changes_from_previous":   changes_from_previous,
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
        "objects_visible":         [],
        "spatial_characteristics": [],
        "changes_from_previous": {
            "camera_movement":      "unknown",
            "new_elements":         [],
            "disappeared_elements": [],
            "significant_change":   False,
            "change_description":   "default perception",
        },
        "confidence":              0.1,
    }
