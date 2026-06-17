"""
Stage 2 — Per-frame semantic perception using Qwen2.5-VL-3B via Ollama.

v4 changes (consistency + anti-hallucination)
---------------------------------------------
1. Deterministic inference: temperature=0.0, seed=42, top_k=1, top_p=1.0
   → identical frames produce byte-identical outputs across runs.

2. Closed-enum leads_to in all prompts:
   "entrance|corridor|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown"
   → model cannot invent room types it hasn't physically seen.

3. One-shot JSON example embedded in every prompt:
   → gives the model a concrete format anchor; reduces schema drift.

4. Strict verify pass: _verify_perception() now rejects any confirmed_room_type
   not in VALID_ROOM_TYPES (falls back to original rather than adopting garbage).

5. Tiebreaker pass: if first and second passes still disagree on room_type,
   a minimal _PROMPT_TIEBREAK forces a binary choice between the two candidates.
   This eliminates split-personality frames.

6. Hallucination guard on leads_to: if a door's leads_to is not confirmed in at
   least 1 adjacent frame within a 3-frame window, its confidence is clamped to
   min(original, 0.35) so Stage 3's ghost logic deprioritises it.
   (Applied by Stage 2 Audit, not here.)

Door schema (unchanged from v3)
---------------------------------
doors_visible is a list of rich door objects:
  [
    {"side": "left",  "open": true,  "leads_to": "corridor", "confidence": 0.85},
    {"side": "right", "open": false, "leads_to": "unknown",  "confidence": 0.70}
  ]

Two primary prompts:
  _PROMPT_FIRST     — first frame only; no previous frame to compare.
  _PROMPT_WITH_PREV — all subsequent frames; prev+curr stitched side-by-side.
  _PROMPT_VERIFY    — re-query to validate a prior analysis.
  _PROMPT_TIEBREAK  — forced binary choice when first two passes disagree.
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

# ── deterministic inference options ──────────────────────────────────────────
# temperature=0.0 + seed=42 + top_k=1 → fully greedy, reproducible outputs.
_INFER_OPTS = {
    "temperature": 0.0,
    "seed":        42,
    "top_k":       1,
    "top_p":       1.0,
    "num_predict": 700,
}

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

# ── closed enum for leads_to (used in all prompts) ──────────────────────────
_LEADS_TO_ENUM = (
    "entrance|corridor|living_room|bedroom|kitchen|"
    "bathroom|office|stairwell|lobby|unknown"
)

# ── one-shot example (compact) embedded in prompts ───────────────────────────
_ONE_SHOT_EXAMPLE = """\
EXAMPLE — output exactly this structure (values will differ for your image):
{
  "room_type": "corridor",
  "alternative_type": null,
  "size_hint": "small",
  "doors_visible": [
    {"side": "front", "open": true,  "leads_to": "living_room", "confidence": 0.80},
    {"side": "left",  "open": false, "leads_to": "bedroom",     "confidence": 0.65}
  ],
  "view_description": "narrow corridor with white walls tiled floor and two doors visible",
  "objects_visible": ["door", "light fixture", "baseboard"],
  "spatial_characteristics": ["tiled floor", "white walls", "fluorescent lighting"],
  "changes_from_previous": {
    "camera_movement": "forward",
    "new_elements": ["second door on left"],
    "disappeared_elements": [],
    "significant_change": false,
    "change_description": "camera moved forward slightly revealing a second door"
  },
  "confidence": 0.82
}"""

# ── prompts ───────────────────────────────────────────────────────────────────

_PROMPT_FIRST = f"""\
You are analyzing the first frame of an indoor walkthrough video.
Describe in careful detail ONLY what is directly visible in the image.
Do NOT guess or infer rooms that are not physically visible.

Output ONLY valid JSON — no markdown fences, no explanation:
{{
  "room_type": "entrance|corridor|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown",
  "alternative_type": "second-best room type or null if confident",
  "size_hint": "very_small|small|medium|large|very_large",
  "doors_visible": [
    {{
      "side": "left|right|front|none",
      "open": true,
      "leads_to": "{_LEADS_TO_ENUM}",
      "confidence": 0.0
    }}
  ],
  "view_description": "detailed 15-25 word description of what is literally visible",
  "objects_visible": ["every distinct object you can identify"],
  "spatial_characteristics": ["floor material", "wall material", "ceiling detail", "lighting"],
  "changes_from_previous": {{
    "camera_movement": "unknown",
    "new_elements": [],
    "disappeared_elements": [],
    "significant_change": false,
    "change_description": "first frame, no previous to compare"
  }},
  "confidence": 0.0
}}

{_ONE_SHOT_EXAMPLE}

Rules (STRICT — do not violate):
- ONLY describe what you can literally see. Do NOT hallucinate unseen areas.
- doors_visible: list EVERY door you can see. If no doors visible, use [{{"side":"none","open":false,"leads_to":"unknown","confidence":0.0}}].
- leads_to MUST be one of: {_LEADS_TO_ENUM}
- For each door: "side" is which wall it is on. "open"=true only if door is open or ajar.
- leads_to: only guess the adjacent room if part of it is actually visible through the door. Otherwise use "unknown".
- confidence: 0.8+ if the room type is unmistakable, 0.5-0.7 if partial view, <0.5 if unclear.
- If confidence < 0.65 set alternative_type to the second-best guess; otherwise null."""

_PROMPT_WITH_PREV = f"""\
This image shows TWO consecutive indoor video frames placed side by side.
LEFT HALF = PREVIOUS frame.  RIGHT HALF = CURRENT frame.

TASK 1: Analyze the CURRENT frame (right half) in full detail.
TASK 2: Compare CURRENT to PREVIOUS and describe exactly what changed.

Output ONLY valid JSON — no markdown fences, no explanation:
{{
  "room_type": "entrance|corridor|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown",
  "alternative_type": "second-best room type or null if confident",
  "size_hint": "very_small|small|medium|large|very_large",
  "doors_visible": [
    {{
      "side": "left|right|front|none",
      "open": true,
      "leads_to": "{_LEADS_TO_ENUM}",
      "confidence": 0.0
    }}
  ],
  "view_description": "detailed 15-25 word description of the CURRENT frame (right half)",
  "objects_visible": ["every distinct object visible in the CURRENT frame"],
  "spatial_characteristics": ["floor material", "wall material", "ceiling detail", "lighting"],
  "changes_from_previous": {{
    "camera_movement": "forward|backward|left|right|rotating_left|rotating_right|static",
    "new_elements": ["things that appeared in CURRENT not visible in PREVIOUS"],
    "disappeared_elements": ["things in PREVIOUS that are gone in CURRENT"],
    "significant_change": false,
    "change_description": "1-2 sentences on the most important change between frames"
  }},
  "confidence": 0.0
}}

{_ONE_SHOT_EXAMPLE}

Rules (STRICT — do not violate):
- room_type, view_description, objects_visible, spatial_characteristics, doors_visible: CURRENT frame (right half) ONLY.
- changes_from_previous: compare CURRENT (right) to PREVIOUS (left).
- ONLY describe what you can literally see. Do NOT invent rooms not visible.
- doors_visible: list EVERY door you see in CURRENT frame. If none, use [{{"side":"none","open":false,"leads_to":"unknown","confidence":0.0}}].
- leads_to MUST be one of: {_LEADS_TO_ENUM}
- leads_to: only guess the adjacent room type if part of it is actually visible through the open door. Otherwise use "unknown".
- significant_change=true ONLY if a new room became visible, a door appeared/disappeared, or a dramatic viewpoint shift occurred.
- confidence: 0.8+ if room type unmistakable, 0.5-0.7 partial view, <0.5 unclear.
- If confidence < 0.65 set alternative_type; otherwise null."""

_PROMPT_VERIFY = f"""\
You are verifying a prior analysis of this indoor frame.

Prior analysis:
{{prior_json}}

Look at the image carefully and answer these two questions:
1. Is the room_type "{{room_type}}" correct for what is visible?
2. Which of the doors listed below are you CERTAIN actually exist in this image?

Output ONLY valid JSON — no markdown fences, no explanation:
{{{{
  "room_type_confirmed": true,
  "confirmed_room_type": "entrance|corridor|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown",
  "confirmed_doors": [
    {{{{
      "side": "left|right|front",
      "open": true,
      "leads_to": "{_LEADS_TO_ENUM}",
      "confidence": 0.0
    }}}}
  ],
  "notes": "brief explanation of any corrections"
}}}}

Rules (STRICT):
- confirmed_room_type MUST be one of: entrance|corridor|living_room|bedroom|kitchen|bathroom|office|stairwell|lobby|unknown
- confirmed_doors: ONLY list doors you can clearly see in the image. Remove any that do not exist.
- If no doors are visible, return confirmed_doors: []
- leads_to MUST be one of: {_LEADS_TO_ENUM}
- Do NOT guess leads_to unless you can see into the room through an open door."""

_PROMPT_TIEBREAK = """\
Two analyses of the same indoor image produced different room type labels:
  Analysis A says: "{type_a}"
  Analysis B says: "{type_b}"

Look at this image again very carefully.
Choose EXACTLY ONE of the two options — do not introduce a third option.

Output ONLY valid JSON — no markdown fences, no explanation:
{{
  "chosen_type": "{type_a}",
  "reason": "one sentence explaining why"
}}

Rules:
- chosen_type MUST be exactly "{type_a}" or "{type_b}" — nothing else."""


# ── public API ────────────────────────────────────────────────────────────────

def analyze_frames(frame_paths: list) -> list:
    """
    Run Qwen2.5-VL via Ollama on every frame with temporal comparison,
    consistency re-query, and (where needed) a tiebreaker pass.
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
        doors   = perc.get("doors_visible", [])
        door_sides = [d["side"] for d in doors if d.get("side") != "none"]
        verified_str = " [verified]" if perc.get("_verified") else ""
        tiebreak_str = " [tiebreak]" if perc.get("_tiebroken") else ""
        print(
            f"  -> {perc['room_type']:<20}{alt_str}"
            f"  size={perc['size_hint']:<12}"
            f"  move={delta.get('camera_movement', '?'):<16}"
            f"  sig={str(delta.get('significant_change', False)):<6}"
            f"  conf={perc['confidence']:.2f}"
            f"  doors={door_sides}{verified_str}{tiebreak_str}"
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
    Each half is capped at _MAX_IMG_DIM × _MAX_IMG_DIM before compositing.
    """
    target_h = _MAX_IMG_DIM

    def _resize_h(img: Image.Image, h: int) -> Image.Image:
        scale = h / img.height
        w = max(1, int(img.width * scale))
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

def _call_ollama(prompt: str, b64: str, retries: int = 2,
                 num_predict: int = 700) -> str | None:
    opts = {**_INFER_OPTS, "num_predict": num_predict}
    for attempt in range(retries):
        try:
            resp = requests.post(
                _OLLAMA_URL,
                json={
                    "model":   _MODEL,
                    "prompt":  prompt,
                    "images":  [b64],
                    "stream":  False,
                    "options": opts,
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
        curr_b64   = _image_to_b64(curr_image)

        if prev_image is None:
            b64 = curr_b64
            raw = _call_ollama(_PROMPT_FIRST, b64)
        else:
            b64 = _make_side_by_side(prev_image, curr_image)
            raw = _call_ollama(_PROMPT_WITH_PREV, b64)

        if raw is None:
            return _default_perception(frame_id)

        perc = _parse_response(raw, frame_id)

        # ── Consistency re-query pass ─────────────────────────────────────────
        # Trigger if: confidence is uncertain OR any real door was detected.
        real_doors = [d for d in perc.get("doors_visible", [])
                      if d.get("side") != "none"]
        should_verify = perc["confidence"] < 0.65 or len(real_doors) > 0

        if should_verify:
            perc = _verify_perception(perc, curr_b64, frame_id)

        return perc

    except Exception as exc:
        print(f"\n  WARNING: frame {frame_id} failed ({exc})")
        return _default_perception(frame_id)


def _verify_perception(perc: dict, curr_b64: str, frame_id: int) -> dict:
    """
    Second-pass consistency check: re-query Qwen2.5-VL with the same current
    frame image and ask it to confirm door detections and room type.
    Only doors confirmed in both passes are kept.
    If room_type still disagrees, run a tiebreaker pass.
    """
    prior_summary = {
        "room_type":    perc["room_type"],
        "doors_visible": perc.get("doors_visible", []),
        "confidence":   perc["confidence"],
    }
    prior_json_str = json.dumps(prior_summary, indent=2)
    prompt = _PROMPT_VERIFY.format(
        prior_json=prior_json_str,
        room_type=perc["room_type"],
    )

    raw = _call_ollama(prompt, curr_b64)
    if raw is None:
        return perc  # keep original if re-query fails

    # Parse verification response
    raw = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "")
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return perc  # unparseable → keep original

    try:
        v = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return perc

    # Merge results
    merged = dict(perc)
    merged["_verified"] = True

    # Room type: only adopt corrected type if it is in VALID_ROOM_TYPES
    if not v.get("room_type_confirmed", True):
        raw_corrected = str(v.get("confirmed_room_type", perc["room_type"]))
        corrected_rt  = _normalize_room_type(raw_corrected)

        if corrected_rt not in VALID_ROOM_TYPES or corrected_rt == "unknown":
            # Verification produced garbage — keep original
            pass
        elif corrected_rt != perc["room_type"]:
            # Disagreement between pass 1 and pass 2 — run tiebreaker
            resolved = _tiebreak_room_type(
                perc["room_type"], corrected_rt, curr_b64
            )
            if resolved != perc["room_type"]:
                merged["alternative_type"] = perc["room_type"]
                merged["room_type"]        = resolved
                merged["confidence"]       = round(merged["confidence"] * 0.85, 2)
                merged["_tiebroken"]       = True
            else:
                merged["_tiebroken"] = False
        else:
            merged["_tiebroken"] = False
    else:
        merged["_tiebroken"] = False

    # Doors: intersect — keep only doors confirmed by re-query
    confirmed     = _parse_door_list(v.get("confirmed_doors", []))
    confirmed_sides = {d["side"] for d in confirmed if d["side"] != "none"}

    prior_doors = perc.get("doors_visible", [])
    kept: list[dict] = []
    for d in prior_doors:
        if d["side"] == "none":
            continue
        if d["side"] in confirmed_sides:
            # Blend confidence across both passes
            for cd in confirmed:
                if cd["side"] == d["side"]:
                    blended_conf = round((d["confidence"] + cd["confidence"]) / 2, 2)
                    merged_door  = {**d, "confidence": blended_conf}
                    # Prefer more specific leads_to from verification
                    if cd.get("leads_to") and cd["leads_to"] != "unknown":
                        merged_door["leads_to"] = cd["leads_to"]
                    kept.append(merged_door)
                    break
            else:
                kept.append(d)

    if not kept:
        kept = [{"side": "none", "open": False, "leads_to": "unknown", "confidence": 0.0}]

    merged["doors_visible"] = kept
    return merged


def _tiebreak_room_type(type_a: str, type_b: str, curr_b64: str) -> str:
    """
    When pass 1 says type_a and pass 2 says type_b, run a minimal forced-choice
    prompt that makes the model pick exactly one of the two. Returns the winner.
    Falls back to type_a (pass 1) if the call fails or returns garbage.
    """
    prompt = _PROMPT_TIEBREAK.format(type_a=type_a, type_b=type_b)
    raw = _call_ollama(prompt, curr_b64, num_predict=150)
    if raw is None:
        return type_a

    raw = re.sub(r"```(?:json)?", "", raw).strip().replace("```", "")
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        return type_a

    try:
        tb = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return type_a

    chosen = str(tb.get("chosen_type", type_a)).strip().lower().replace(" ", "_")
    # Must be exactly one of the two candidates
    if chosen == type_b:
        return type_b
    return type_a  # default to pass-1 answer


# ── JSON parsing & normalisation ──────────────────────────────────────────────

def _normalize_room_type(rt: str) -> str:
    rt = rt.strip().lower().replace(" ", "_")
    rt = SEMANTIC_CANONICAL.get(rt, rt)
    return rt if rt in VALID_ROOM_TYPES else "unknown"


def _clean_str_list(lst, max_n: int) -> list:
    if not isinstance(lst, list):
        return []
    return [str(e).strip() for e in lst if str(e).strip()][:max_n]


def _parse_door_list(raw_doors) -> list[dict]:
    """
    Parse the doors_visible field from VLM JSON.
    Handles both the new object format and legacy flat-string format.
    Returns a list of normalised door dicts.
    leads_to is always validated against VALID_ROOM_TYPES.
    """
    if not isinstance(raw_doors, list):
        return [{"side": "none", "open": False, "leads_to": "unknown", "confidence": 0.0}]

    result: list[dict] = []
    for entry in raw_doors:
        if isinstance(entry, str):
            side = entry.lower().strip()
            if side not in VALID_DOOR_SIDES:
                side = "none"
            result.append({"side": side, "open": False, "leads_to": "unknown", "confidence": 0.5})

        elif isinstance(entry, dict):
            side = str(entry.get("side", "none")).lower().strip()
            if side not in VALID_DOOR_SIDES:
                side = "none"

            open_val = entry.get("open", False)
            if isinstance(open_val, str):
                open_val = open_val.strip().lower() in ("true", "1", "yes")
            else:
                open_val = bool(open_val)

            leads_to_raw = str(entry.get("leads_to", "unknown")).strip().lower().replace(" ", "_")
            leads_to     = _normalize_room_type(leads_to_raw)
            # If normalization produced "unknown" from a non-"unknown" input, keep "unknown"
            if leads_to not in VALID_ROOM_TYPES:
                leads_to = "unknown"

            try:
                conf = round(max(0.0, min(1.0, float(entry.get("confidence", 0.5)))), 2)
            except (TypeError, ValueError):
                conf = 0.5

            result.append({"side": side, "open": open_val, "leads_to": leads_to, "confidence": conf})

    if not result:
        result = [{"side": "none", "open": False, "leads_to": "unknown", "confidence": 0.0}]

    return result


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

    # --- doors_visible (new rich schema) ---
    doors_visible = _parse_door_list(raw.get("doors_visible", []))

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
        "_verified":               False,
        "_tiebroken":              False,
    }


def _default_perception(frame_id: int) -> dict:
    return {
        "frame_id":                frame_id,
        "room_type":               "unknown",
        "alternative_type":        None,
        "size_hint":               "medium",
        "doors_visible":           [{"side": "none", "open": False, "leads_to": "unknown", "confidence": 0.0}],
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
        "_verified":               False,
        "_tiebroken":              False,
    }
