import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import gc
import json
import uuid
import traceback
import textwrap
import numpy as np
from PIL import Image, ImageDraw

import cv2
import torch
import torch.nn.functional as F
import gradio as gr

from scipy.optimize import linear_sum_assignment

from transformers import (
    AutoProcessor,
    AutoImageProcessor,
    AutoModel,
    Qwen2VLForConditionalGeneration,
    GroundingDinoProcessor,
    GroundingDinoForObjectDetection,
)

from qwen_vl_utils import process_vision_info

# =========================================================
# 0. Config
# =========================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if HAS_CUDA else "cpu"
DTYPE = torch.float16 if HAS_CUDA else torch.float32

OUTPUT_DIR = "outputs_position_based_box_spotdiff"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
DINO_MODEL_ID = os.getenv("DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
DINOV2_MODEL_ID = os.getenv("DINOV2_MODEL_ID", "facebook/dinov2-base")

MAX_TAGS_DEFAULT = 12
MAX_BOXES_PER_TAG = 6
MAX_FINAL_DIFFS = 5

DINO_BOX_THR_DEFAULT = 0.10
DINO_TEXT_THR_DEFAULT = 0.10

MATCH_IOU_THR_DEFAULT = 0.05
EXPANDED_IOU_THR_DEFAULT = 0.10
CENTER_DIST_THR_DEFAULT = 0.28
VERIFY_THRESHOLD_DEFAULT = 0.0

CROP_PAD_RATIO = 0.24

MIN_BOX_AREA_RATIO = 0.0015
MAX_BOX_AREA_RATIO = 0.50
MAX_ASPECT_RATIO = 7.0
MIN_ASPECT_RATIO = 0.15

BOX_NMS_IOU = 0.55
FINAL_DUP_IOU = 0.55

UNMATCHED_SAME_DINO_THR = 0.18
UNMATCHED_SAME_COLOR_THR = 35.0

MIN_MISSING_SCORE = 0.25
MIN_MISSING_AREA_RATIO = 0.004
MAX_NEARBY_IOU_FOR_MISSING = 0.15
MIN_CENTER_FOR_MISSING = 0.12

BOX_COLORS = [
    (255, 0, 0),
    (0, 120, 255),
    (0, 190, 70),
    (255, 160, 0),
    (180, 0, 255),
    (0, 180, 180),
]

print("=" * 80)
print("Position-Based Detected-Box Visual Difference Explainer")
print("Pipeline: Qwen Tags → Grounding DINO Boxes → Position Matching → DINOv2/Color Scoring → Qwen Explanation")
print(f"CUDA: {HAS_CUDA}")
print(f"DEVICE: {DEVICE}")
print(f"QWEN_MODEL_ID: {QWEN_MODEL_ID}")
print(f"DINO_MODEL_ID: {DINO_MODEL_ID}")
print(f"DINOV2_MODEL_ID: {DINOV2_MODEL_ID}")
print("=" * 80)

# =========================================================
# 1. Load Models
# =========================================================
print("[Loading] Qwen2-VL...")
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)
qwen_model.to(DEVICE)
qwen_model.eval()
print("[Done] Qwen2-VL loaded.")

print("[Loading] Grounding DINO...")
dino_processor = GroundingDinoProcessor.from_pretrained(DINO_MODEL_ID)
dino_model = GroundingDinoForObjectDetection.from_pretrained(
    DINO_MODEL_ID,
    torch_dtype=torch.float32,
)
dino_model.to(DEVICE)
dino_model.eval()
print("[Done] Grounding DINO loaded.")

print("[Loading] DINOv2...")
dinov2_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
dinov2_model = AutoModel.from_pretrained(
    DINOV2_MODEL_ID,
    torch_dtype=DTYPE,
)
dinov2_model.to(DEVICE)
dinov2_model.eval()
print("[Done] DINOv2 loaded.")

# =========================================================
# 2. Basic Utils
# =========================================================
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_temp_image(image: Image.Image, prefix="tmp"):
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{uuid.uuid4().hex[:8]}.png")
    image.save(path)
    return path

def ensure_same_size(img_a: Image.Image, img_b: Image.Image):
    img_a = img_a.convert("RGB")
    img_b = img_b.convert("RGB")

    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)

    return img_a, img_b

def make_placeholder_image(text, width=320, height=220):
    img = Image.new("RGB", (width, height), (35, 35, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, width - 10, height - 10], outline=(220, 220, 220), width=2)
    draw.text((20, height // 2 - 10), text, fill=(255, 255, 255))
    return img

def clamp_box(box, w, h):
    if box is None:
        return None

    vals = [float(v) for v in box]
    if len(vals) != 4:
        return None

    x1, y1, x2, y2 = vals

    if not np.isfinite([x1, y1, x2, y2]).all():
        return None

    xa = min(x1, x2)
    xb = max(x1, x2)
    ya = min(y1, y2)
    yb = max(y1, y2)

    xa = max(0, min(w - 1, xa))
    xb = max(0, min(w - 1, xb))
    ya = max(0, min(h - 1, ya))
    yb = max(0, min(h - 1, yb))

    if xb <= xa:
        xb = min(w - 1, xa + 1)
    if yb <= ya:
        yb = min(h - 1, ya + 1)

    return [float(xa), float(ya), float(xb), float(yb)]

def expand_box(box, image_size, pad_ratio=CROP_PAD_RATIO):
    w, h = image_size
    box = clamp_box(box, w, h)

    if box is None:
        return None

    x1, y1, x2, y2 = box

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    px = bw * pad_ratio
    py = bh * pad_ratio

    return clamp_box([x1 - px, y1 - py, x2 + px, y2 + py], w, h)

def expand_box_by_ratio(box, image_size, ratio=0.25):
    if box is None:
        return None

    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in box]

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    px = bw * ratio
    py = bh * ratio

    return clamp_box([x1 - px, y1 - py, x2 + px, y2 + py], w, h)

def crop_from_box(image: Image.Image, box, pad_ratio=CROP_PAD_RATIO):
    if box is None:
        return make_placeholder_image("NO BOX")

    safe = expand_box(box, image.size, pad_ratio)

    if safe is None:
        return make_placeholder_image("INVALID BOX")

    x1, y1, x2, y2 = [int(v) for v in safe]

    if x2 <= x1 or y2 <= y1:
        return make_placeholder_image("INVALID BOX")

    return image.crop((x1, y1, x2, y2))

def box_area(box):
    if box is None:
        return 0.0

    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def box_iou(a, b):
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = box_area(a) + box_area(b) - inter + 1e-6

    return float(inter / union)

def expanded_box_iou(a, b, image_size, ratio=0.25):
    ea = expand_box_by_ratio(a, image_size, ratio)
    eb = expand_box_by_ratio(b, image_size, ratio)
    return box_iou(ea, eb)

def box_center(box):
    if box is None:
        return None

    x1, y1, x2, y2 = [float(v) for v in box]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def box_center_distance_ratio(a, b, image_size):
    if a is None or b is None:
        return 999.0

    w, h = image_size
    ac = box_center(a)
    bc = box_center(b)

    dx = (ac[0] - bc[0]) / max(w, 1)
    dy = (ac[1] - bc[1]) / max(h, 1)

    return float((dx * dx + dy * dy) ** 0.5)

def parse_json_any(text):
    if text is None:
        return None

    s = str(text).strip()
    s = re.sub(r"^```json", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^```", "", s).strip()
    s = re.sub(r"```$", "", s).strip()

    chunks = []

    a = s.find("[")
    b = s.rfind("]")
    if a != -1 and b != -1 and b > a:
        chunks.append(s[a:b + 1])

    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        chunks.append(s[a:b + 1])

    for c in chunks:
        c = re.sub(r",\s*}", "}", c)
        c = re.sub(r",\s*]", "]", c)

        try:
            return json.loads(c)
        except Exception:
            pass

    return None

# =========================================================
# 3. Object Tag Proposal by Qwen
# =========================================================
def normalize_object_name(text):
    if text is None:
        return ""

    t = str(text).lower().strip()
    t = re.sub(r"[^a-z0-9 \-/]", " ", t)
    t = t.replace("/", " ")

    if not t:
        return ""

    remove_words = {
        "red", "blue", "green", "yellow", "pink", "purple", "orange",
        "black", "white", "brown", "gray", "grey", "gold", "golden",
        "small", "large", "big", "tiny", "round", "striped",
        "left", "right", "top", "bottom", "upper", "lower",
        "front", "back", "visual", "region", "object", "part",
        "changed", "different", "same",
    }

    words = [w for w in t.split() if w not in remove_words]
    words = [w for w in words if w]

    if len(words) == 0:
        return ""

    t = " ".join(words).strip()

    return t

def normalize_tags(tags):
    clean = []
    seen = set()

    banned = {
        "",
        "image",
        "picture",
        "photo",
        "scene",
        "background",
        "foreground",
        "thing",
        "things",
        "stuff",
        "area",
        "visual",
        "region",
        "regions",
        "object",
        "objects",
        "item",
        "items",
        "difference",
        "differences",
        "change",
        "changes",
    }

    for t in tags:
        nt = normalize_object_name(t)

        if nt in banned:
            continue

        if len(nt) < 2:
            continue

        if nt not in seen:
            seen.add(nt)
            clean.append(nt)

    return clean

def qwen_extract_object_tags(img_a, img_b, max_tags=12):
    path_a = save_temp_image(img_a, "tag_A")
    path_b = save_temp_image(img_b, "tag_B")

    instruction = f"""
    You are given two images: Image A and Image B.

    Your task is NOT to find differences.
    Your task is only to list object categories that are visible in either image.

    Rules:
    - Return object names only.
    - Use simple canonical nouns suitable for open-vocabulary object detection.
    - Do not include colors, sizes, positions, or adjectives.
    - Do not describe differences.
    - Do not manually merge different objects.
    - Include objects from both images.
    - Include up to {max_tags} object categories.
    - Return ONLY valid JSON list of strings.
    - No markdown.

    Example output:
    ["person", "dog", "chair", "tree"]
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path_a},
                {"type": "image", "image": path_b},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=260,
            do_sample=False,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output = qwen_processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    cleanup()

    parsed = parse_json_any(output)

    tags = []

    if isinstance(parsed, list):
        tags = [str(x) for x in parsed if isinstance(x, (str, int, float))]
    elif isinstance(parsed, dict):
        for key in ["objects", "tags", "object_tags"]:
            if isinstance(parsed.get(key), list):
                tags = [str(x) for x in parsed[key]]
                break

    tags = normalize_tags(tags)

    if len(tags) == 0:
        tags = ["person", "animal", "plant", "fish", "bird", "car", "chair", "table"]

    tags = tags[:max_tags]

    return tags, output

# =========================================================
# 4. Grounding DINO Detection
# =========================================================
def make_dino_prompt(tag):
    tag = normalize_object_name(tag)

    if not tag:
        return "an object."

    article = "an" if tag[0].lower() in "aeiou" else "a"
    return f"{article} {tag}."

def valid_detection_box(box, image_size):
    if box is None:
        return False

    w, h = image_size
    area = box_area(box)
    ratio = area / max(w * h, 1)

    if ratio < MIN_BOX_AREA_RATIO:
        return False

    if ratio > MAX_BOX_AREA_RATIO:
        return False

    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    aspect = bw / bh

    if aspect > MAX_ASPECT_RATIO:
        return False

    if aspect < MIN_ASPECT_RATIO:
        return False

    if bw < 18 or bh < 18:
        return False

    return True

def nms_detections(dets, iou_thr=BOX_NMS_IOU):
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    kept = []

    for d in dets:
        duplicate = False

        for k in kept:
            if box_iou(d["box"], k["box"]) > iou_thr:
                duplicate = True
                break

        if duplicate:
            continue

        kept.append(d)

    return kept

def detect_one_tag(image: Image.Image, tag, box_thr=0.10, text_thr=0.10):
    tag = normalize_object_name(tag)

    if not tag:
        return []

    prompt = make_dino_prompt(tag)

    inputs = dino_processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=DEVICE, dtype=torch.float32)

    with torch.inference_mode():
        outputs = dino_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)

    try:
        results = dino_processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.get("input_ids", None),
            box_threshold=float(box_thr),
            text_threshold=float(text_thr),
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        try:
            results = dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=target_sizes,
                threshold=float(box_thr),
            )[0]
        except TypeError:
            results = dino_processor.post_process_grounded_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=float(box_thr),
            )[0]

    boxes = results.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
    scores = results.get("scores", torch.empty((0,))).detach().cpu().numpy()

    cleanup()

    dets = []

    if len(boxes) == 0:
        return dets

    order = np.argsort(-scores)

    for idx in order:
        box = clamp_box(boxes[idx].tolist(), image.size[0], image.size[1])

        if box is None:
            continue

        if not valid_detection_box(box, image.size):
            continue

        dets.append(
            {
                "tag": tag,
                "prompt": prompt,
                "box": box,
                "score": float(scores[idx]),
            }
        )

        if len(dets) >= MAX_BOXES_PER_TAG:
            break

    return nms_detections(dets, iou_thr=BOX_NMS_IOU)

def detect_all_tags(image, tags, box_thr=0.10, text_thr=0.10):
    all_dets = []

    for tag in tags:
        dets = detect_one_tag(
            image,
            tag,
            box_thr=box_thr,
            text_thr=text_thr,
        )
        all_dets.extend(dets)

    all_dets = nms_detections(all_dets, iou_thr=BOX_NMS_IOU)

    return all_dets

# =========================================================
# 5. DINOv2 Crop Comparison
# =========================================================
def dinov2_feature(image: Image.Image):
    image = image.convert("RGB")

    inputs = dinov2_processor(images=image, return_tensors="pt")

    inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device=DEVICE, dtype=DTYPE)

    with torch.inference_mode():
        outputs = dinov2_model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feat = outputs.pooler_output
    else:
        feat = outputs.last_hidden_state[:, 0]

    feat = F.normalize(feat.float(), dim=-1)

    cleanup()

    return feat.detach().cpu()

def dinov2_distance(crop_a, crop_b):
    fa = dinov2_feature(crop_a)
    fb = dinov2_feature(crop_b)

    sim = float((fa * fb).sum().item())

    return float(1.0 - sim)

def color_difference_score(crop_a, crop_b):
    size = (160, 160)

    a = np.array(crop_a.convert("RGB").resize(size, Image.LANCZOS))
    b = np.array(crop_b.convert("RGB").resize(size, Image.LANCZOS))

    rgb = cv2.absdiff(a, b)
    rgb_score = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())

    hsv_a = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)

    h_a = hsv_a[:, :, 0].astype(np.int16)
    h_b = hsv_b[:, :, 0].astype(np.int16)

    hue_abs = np.abs(h_a - h_b)
    hue_diff = float(np.minimum(hue_abs, 180 - hue_abs).astype(np.float32).mean())

    sat_diff = float(cv2.absdiff(hsv_a[:, :, 1], hsv_b[:, :, 1]).mean())
    val_diff = float(cv2.absdiff(hsv_a[:, :, 2], hsv_b[:, :, 2]).mean())

    return float(0.45 * rgb_score + 0.25 * hue_diff + 0.20 * sat_diff + 0.10 * val_diff)

def compute_visual_evidence(crop_a, crop_b):
    crop_b_flipped = crop_b.transpose(Image.FLIP_LEFT_RIGHT)

    normal_dino = dinov2_distance(crop_a, crop_b)
    flipped_dino = dinov2_distance(crop_a, crop_b_flipped)

    normal_color = color_difference_score(crop_a, crop_b)
    flipped_color = color_difference_score(crop_a, crop_b_flipped)

    return {
        "crop_b_flipped": crop_b_flipped,
        "normal_dino": float(normal_dino),
        "flipped_dino": float(flipped_dino),
        "dino_flip_improvement": float(normal_dino - flipped_dino),
        "normal_color": float(normal_color),
        "flipped_color": float(flipped_color),
        "color_flip_improvement": float(normal_color - flipped_color),
    }

# =========================================================
# 6. Classification / Filtering
# =========================================================

def fallback_classify_pair(pair):
    pair_type = pair.get("pair_type", "matched")
    dino = float(pair.get("dinov2_dist", 0.0))
    color = float(pair.get("color_score", 0.0))
    iou = float(pair.get("bbox_iou", 0.0))
    expanded_iou = float(pair.get("expanded_iou", 0.0))
    center = float(pair.get("center_dist", 0.0))

    dino_flip_imp = float(pair.get("dino_flip_improvement", 0.0))
    color_flip_imp = float(pair.get("color_flip_improvement", 0.0))

    same_tag = pair.get("tag_a", "") == pair.get("tag_b", "")

    if pair_type == "missing_in_B":
        return "missing_in_B", 5

    if pair_type == "added_in_B":
        return "added_in_B", 5

    if iou >= 0.80 and center <= 0.04 and dino < 0.10 and color < 16.0:
        return "minor_or_same", 1

    if same_tag and iou >= 0.70 and center <= 0.06 and color < 12.0 and dino < 0.25:
        return "minor_or_same", 1

    if same_tag and expanded_iou >= 0.45 and dino < 0.10 and color < 16.0:
        return "minor_or_same", 1

    if same_tag and expanded_iou >= 0.25 and (
        dino_flip_imp >= 0.020 or color_flip_imp >= 6.0
    ):
        return "orientation_changed", 5

    if same_tag and expanded_iou >= 0.25 and color >= 18.0 and dino < 0.22:
        return "color_changed", 5

    if pair_type == "matched_different_label":
        if dino >= 0.16 or color >= 18.0:
            return "object_or_region_changed", 5
        return "minor_or_same", 1

    if dino >= 0.20 and color >= 18.0:
        return "object_or_region_changed", 5

    if dino >= 0.16:
        return "appearance_changed", 4

    if color >= 20.0:
        return "color_changed", 4

    if center > 0.16 and (dino >= 0.10 or color >= 18.0):
        return "moved_or_changed", 4

    return "minor_or_same", 1


def compute_change_score(pair):
    pair_type = pair.get("pair_type", "matched")
    change_type = pair.get("change_type", "")

    dino = float(pair.get("dinov2_dist", 0.0))
    dino_flip_imp = float(pair.get("dino_flip_improvement", 0.0))
    color = float(pair.get("color_score", 0.0))
    color_flip_imp = float(pair.get("color_flip_improvement", 0.0))
    expanded_iou = float(pair.get("expanded_iou", 0.0))
    center = float(pair.get("center_dist", 0.0))

    if pair_type in ["missing_in_B", "added_in_B"]:
        return 35.0

    low_overlap_penalty = 0.0
    if expanded_iou < 0.10:
        low_overlap_penalty = 10.0

    if change_type == "orientation_changed":
        return (
            max(0.0, dino_flip_imp) * 180.0
            + max(0.0, color_flip_imp) * 1.5
            + dino * 45.0
            + color * 0.35
            + expanded_iou * 12.0
            - low_overlap_penalty
        )

    if change_type == "object_or_region_changed":
        return dino * 100.0 + color * 0.6 + center * 20.0 + expanded_iou * 8.0 - low_overlap_penalty

    if change_type == "color_changed":
        return color * 1.15 + dino * 35.0 + expanded_iou * 12.0 - low_overlap_penalty

    if change_type == "appearance_changed":
        return dino * 120.0 + color * 0.45 + expanded_iou * 8.0 - low_overlap_penalty

    if change_type == "moved_or_changed":
        return center * 40.0 + dino * 60.0 + color * 0.35 + expanded_iou * 8.0 - low_overlap_penalty

    if change_type == "region_changed":
        return color * 0.65 + dino * 40.0 + expanded_iou * 8.0 - low_overlap_penalty

    return dino * 50.0 + color * 0.2 - low_overlap_penalty


def compute_candidate_score(pair):
    pair_type = pair.get("pair_type", "matched")

    if pair_type in ["missing_in_B", "added_in_B"]:
        return 35.0

    dino = float(pair.get("dinov2_dist", 0.0))
    color = float(pair.get("color_score", 0.0))
    expanded_iou = float(pair.get("expanded_iou", 0.0))
    center = float(pair.get("center_dist", 0.0))
    dino_flip_imp = max(0.0, float(pair.get("dino_flip_improvement", 0.0)))
    color_flip_imp = max(0.0, float(pair.get("color_flip_improvement", 0.0)))

    return float(
        dino * 100.0
        + color * 0.45
        + expanded_iou * 6.0
        + center * 8.0
        + dino_flip_imp * 80.0
        + color_flip_imp * 0.9
    )

def fallback_description_for_pair(pair):
    tag = pair.get("tag", "object")
    pair_type = pair.get("pair_type", "matched")
    change_type = pair.get("change_type", "region_changed")

    if pair_type == "missing_in_B":
        return f"The {tag} appears in Image A but is missing in Image B."

    if pair_type == "added_in_B":
        return f"The {tag} appears in Image B but not in Image A."

    if change_type == "color_changed":
        return f"The {tag} changes color between Image A and Image B."

    if change_type == "appearance_changed":
        return f"The {tag} changes in appearance between Image A and Image B."

    if change_type == "object_or_region_changed":
        return "The selected region changes into a different object-like region between Image A and Image B."

    if change_type == "orientation_changed":
        return f"The {tag} appears with a different orientation between Image A and Image B."

    if change_type == "flipped":
        return f"The {tag} appears horizontally flipped between Image A and Image B."

    if change_type == "moved_or_changed":
        return f"The {tag} appears shifted or visually changed between Image A and Image B."

    return "The selected region shows a visible difference between Image A and Image B."


def veto_qwen_judgement(pair):
    pair_type = pair.get("pair_type", "matched")

    if pair_type in ["missing_in_B", "added_in_B"]:
        return pair

    tag_a = pair.get("tag_a", "")
    tag_b = pair.get("tag_b", "")
    same_tag = tag_a == tag_b

    iou = float(pair.get("bbox_iou", 0.0))
    center = float(pair.get("center_dist", 0.0))
    dino = float(pair.get("dinov2_dist", 0.0))
    color = float(pair.get("color_score", 0.0))

    if iou >= 0.80 and center <= 0.04 and dino < 0.12 and color < 18.0:
        pair["change_type"] = "minor_or_same"
        pair["importance"] = 1
        pair["confidence"] = "filtered_by_evidence"
        pair["description"] = "The matched crops do not show a meaningful visual difference."
        return pair

    if same_tag and iou >= 0.70 and center <= 0.06 and color < 12.0 and dino < 0.25:
        pair["change_type"] = "minor_or_same"
        pair["importance"] = 1
        pair["confidence"] = "filtered_by_evidence"
        pair["description"] = "The matched object appears visually similar in both images."
        return pair

    if same_tag and iou >= 0.65 and dino < 0.10 and color < 18.0:
        pair["change_type"] = "minor_or_same"
        pair["importance"] = 1
        pair["confidence"] = "filtered_by_evidence"
        pair["description"] = "The matched object appears visually similar in both images."
        return pair

    if (not same_tag) and iou >= 0.65 and center <= 0.06 and color < 14.0 and dino < 0.16:
        pair["change_type"] = "minor_or_same"
        pair["importance"] = 1
        pair["confidence"] = "filtered_by_evidence"
        pair["description"] = "The detected labels differ, but the crop evidence looks visually similar."
        return pair

    return pair


# =========================================================
# 7. Position-Based Global Box Matching
# =========================================================
def matching_cost(det_a, det_b, image_size):
    iou = box_iou(det_a["box"], det_b["box"])
    expanded_iou = expanded_box_iou(det_a["box"], det_b["box"], image_size, ratio=0.30)
    center_dist = box_center_distance_ratio(det_a["box"], det_b["box"], image_size)

    area_a = box_area(det_a["box"])
    area_b = box_area(det_b["box"])
    area_ratio = abs(area_a - area_b) / max(area_a, area_b, 1.0)

    score_gap = abs(float(det_a.get("score", 0.0)) - float(det_b.get("score", 0.0)))

    cost = (
        (1.0 - expanded_iou) * 3.0
        + (1.0 - iou) * 1.2
        + area_ratio * 1.8
        + center_dist * 0.8
        + score_gap * 0.1
    )

    return float(cost), float(iou), float(expanded_iou), float(center_dist), float(area_ratio)


def add_if_not_duplicate(final, p):
    p_type = str(p.get("change_type", ""))
    p_tag = str(p.get("tag", ""))

    for q in final:
        q_type = str(q.get("change_type", ""))
        q_tag = str(q.get("tag", ""))

        overlap_a = box_iou(p.get("box_a"), q.get("box_a"))
        overlap_b = box_iou(p.get("box_b"), q.get("box_b"))

        both_side_overlap = overlap_a > 0.72 and overlap_b > 0.72
        one_side_overlap = max(overlap_a, overlap_b) > 0.88

        if p_tag == q_tag and p_type == q_type:
            if both_side_overlap or one_side_overlap:
                return False

    final.append(p)
    return True


def select_diverse_final(results):
    results = [
        p for p in results
        if p.get("change_type") != "minor_or_same"
    ]

    for p in results:
        if "change_score" not in p:
            p["change_score"] = compute_change_score(p)

    results = sorted(
        results,
        key=lambda p: (
            int(p.get("importance", 1)),
            float(p.get("change_score", 0.0)),
            float(p.get("color_score", 0.0)),
        ),
        reverse=True,
    )

    final = []

    priority_types = [
        "orientation_changed",
        "object_or_region_changed",
        "color_changed",
        "missing_in_B",
        "added_in_B",
        "moved_or_changed",
    ]

    for t in priority_types:
        same_type = [p for p in results if p.get("change_type") == t]
        same_type = sorted(
            same_type,
            key=lambda p: float(p.get("change_score", 0.0)),
            reverse=True,
        )

        for p in same_type:
            if add_if_not_duplicate(final, p):
                break

    for p in results:
        if len(final) >= MAX_FINAL_DIFFS:
            break

        if p.get("change_type") == "appearance_changed":
            dino = float(p.get("dinov2_dist", 0.0))
            color = float(p.get("color_score", 0.0))
            iou = float(p.get("bbox_iou", 0.0))
            center = float(p.get("center_dist", 0.0))

            if iou >= 0.70 and center <= 0.06 and color < 12.0 and dino < 0.25:
                continue

        add_if_not_duplicate(final, p)

    final = final[:MAX_FINAL_DIFFS]

    type_rank = {
        "orientation_changed": 0,
        "object_or_region_changed": 1,
        "color_changed": 2,
        "missing_in_B": 3,
        "added_in_B": 4,
        "moved_or_changed": 5,
        "appearance_changed": 6,
    }

    final = sorted(
        final,
        key=lambda p: (
            type_rank.get(p.get("change_type", ""), 99),
            -float(p.get("change_score", 0.0)),
        ),
    )

    for i, p in enumerate(final):
        p["idx"] = i + 1

    return final


def match_boxes_and_score(
    img_a,
    img_b,
    dets_a,
    dets_b,
    match_iou_thr=0.05,
    center_dist_thr=0.28,
    verify_threshold=0.0,
    enable_missing_added=True,
):
    image_size = img_a.size

    results = []
    used_a = set()
    used_b = set()

    n = len(dets_a)
    m = len(dets_b)

    if n > 0 and m > 0:
        cost_matrix = np.full((n, m), 9999.0, dtype=np.float32)
        info = {}

        for ia, da in enumerate(dets_a):
            for ib, db in enumerate(dets_b):
                cost, iou, expanded_iou, center_dist, area_ratio = matching_cost(
                    da, db, image_size
                )
                info[(ia, ib)] = (cost, iou, expanded_iou, center_dist, area_ratio)

                same_tag = da.get("tag", "") == db.get("tag", "")

                if same_tag:
                    if area_ratio <= 0.85 and (
                        iou >= 0.01
                        or expanded_iou >= 0.05
                        or center_dist <= float(center_dist_thr)
                    ):
                        cost_matrix[ia, ib] = cost
                else:
                    if area_ratio <= 0.80 and (
                        iou >= max(float(match_iou_thr), 0.12)
                        or (expanded_iou >= 0.18 and center_dist <= 0.12)
                    ):
                        cost_matrix[ia, ib] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for ia, ib in zip(row_ind, col_ind):
            if cost_matrix[ia, ib] >= 9999.0:
                continue

            cost, iou, expanded_iou, center_dist, area_ratio = info[(ia, ib)]

            da = dets_a[ia]
            db = dets_b[ib]

            tag_a = da.get("tag", "object")
            tag_b = db.get("tag", "object")
            same_tag = tag_a == tag_b

            if same_tag:
                if area_ratio > 0.85:
                    continue
                if not (
                    iou >= 0.01
                    or expanded_iou >= 0.05
                    or center_dist <= float(center_dist_thr)
                ):
                    continue
            else:
                if area_ratio > 0.80:
                    continue
                if not (
                    iou >= max(float(match_iou_thr), 0.12)
                    or (expanded_iou >= 0.18 and center_dist <= 0.12)
                ):
                    continue

            used_a.add(ia)
            used_b.add(ib)

            crop_a = crop_from_box(img_a, da["box"])
            crop_b = crop_from_box(img_b, db["box"])

            visual_evidence = compute_visual_evidence(crop_a, crop_b)

            dino_dist = visual_evidence["normal_dino"]
            color_score = visual_evidence["normal_color"]

            if tag_a == tag_b:
                tag = tag_a
                pair_type = "matched"
            else:
                tag = f"{tag_a} → {tag_b}"
                pair_type = "matched_different_label"

            pair = {
                "tag": tag,
                "tag_a": tag_a,
                "tag_b": tag_b,
                "pair_type": pair_type,
                "box_a": da["box"],
                "box_b": db["box"],
                "crop_a": crop_a,
                "crop_b": crop_b,
                "score_a": float(da.get("score", 0.0)),
                "score_b": float(db.get("score", 0.0)),
                "bbox_iou": float(iou),
                "expanded_iou": float(expanded_iou),
                "center_dist": float(center_dist),
                "area_ratio": float(area_ratio),
                "match_cost": float(cost),
                "dinov2_dist": float(dino_dist),
                "color_score": float(color_score),
                "crop_b_flipped": visual_evidence["crop_b_flipped"],
                "flipped_dino": float(visual_evidence["flipped_dino"]),
                "flipped_color": float(visual_evidence["flipped_color"]),
                "dino_flip_improvement": float(visual_evidence["dino_flip_improvement"]),
                "color_flip_improvement": float(visual_evidence["color_flip_improvement"]),
                "normal_edge": float(visual_evidence.get("normal_edge", 0.0)),
                "flipped_edge": float(visual_evidence.get("flipped_edge", 0.0)),
                "edge_flip_improvement": float(visual_evidence.get("edge_flip_improvement", 0.0)),
            }

            fallback_change_type, fallback_importance = fallback_classify_pair(pair)

            pair["fallback_change_type"] = fallback_change_type
            pair["fallback_importance"] = fallback_importance
            pair["change_type"] = fallback_change_type
            pair["importance"] = fallback_importance
            pair["confidence"] = "pre_qwen"
            pair["description"] = ""
            pair["change_score"] = compute_candidate_score(pair)

            if pair["change_type"] == "minor_or_same":
                continue

            if pair["change_score"] < float(verify_threshold):
                continue

            results.append(pair)

    if enable_missing_added:
        for ia, da in enumerate(dets_a):
            if ia in used_a:
                continue

            total_area = max(img_a.size[0] * img_a.size[1], 1)
            area_ratio = box_area(da["box"]) / total_area

            if float(da.get("score", 0.0)) < MIN_MISSING_SCORE:
                continue

            if area_ratio < MIN_MISSING_AREA_RATIO:
                continue

            same_loc_crop_a = crop_from_box(img_a, da["box"])
            same_loc_crop_b = crop_from_box(img_b, da["box"])

            same_loc_evidence = compute_visual_evidence(
                same_loc_crop_a,
                same_loc_crop_b,
            )

            if (
                same_loc_evidence["normal_dino"] < UNMATCHED_SAME_DINO_THR
                and same_loc_evidence["normal_color"] < UNMATCHED_SAME_COLOR_THR
            ):
                continue

            pair = {
                "tag": da.get("tag", "object"),
                "tag_a": da.get("tag", "object"),
                "tag_b": "",
                "pair_type": "missing_in_B",
                "box_a": da["box"],
                "box_b": None,
                "crop_a": crop_from_box(img_a, da["box"]),
                "crop_b": make_placeholder_image("MISSING IN B"),
                "same_location_crop_b": same_loc_crop_b,
                "crop_b_flipped": make_placeholder_image("MISSING IN B"),
                "score_a": float(da.get("score", 0.0)),
                "score_b": 0.0,
                "bbox_iou": 0.0,
                "expanded_iou": 0.0,
                "center_dist": 1.0,
                "area_ratio": 1.0,
                "match_cost": 999.0,
                "dinov2_dist": float(same_loc_evidence["normal_dino"]),
                "color_score": float(same_loc_evidence["normal_color"]),
                "flipped_dino": float(same_loc_evidence["flipped_dino"]),
                "flipped_color": float(same_loc_evidence["flipped_color"]),
                "dino_flip_improvement": 0.0,
                "color_flip_improvement": 0.0,
                "normal_edge": 0.0,
                "flipped_edge": 0.0,
                "edge_flip_improvement": 0.0,
                "change_type": "missing_in_B",
                "fallback_change_type": "missing_in_B",
                "fallback_importance": 5,
                "importance": 5,
                "confidence": "pre_qwen",
                "description": "",
                "change_score": 35.0,
            }

            results.append(pair)

        for ib, db in enumerate(dets_b):
            if ib in used_b:
                continue

            total_area = max(img_a.size[0] * img_a.size[1], 1)
            area_ratio = box_area(db["box"]) / total_area

            if float(db.get("score", 0.0)) < MIN_MISSING_SCORE:
                continue

            if area_ratio < MIN_MISSING_AREA_RATIO:
                continue

            same_loc_crop_a = crop_from_box(img_a, db["box"])
            same_loc_crop_b = crop_from_box(img_b, db["box"])

            same_loc_evidence = compute_visual_evidence(
                same_loc_crop_a,
                same_loc_crop_b,
            )

            if (
                same_loc_evidence["normal_dino"] < UNMATCHED_SAME_DINO_THR
                and same_loc_evidence["normal_color"] < UNMATCHED_SAME_COLOR_THR
            ):
                continue

            pair = {
                "tag": db.get("tag", "object"),
                "tag_a": "",
                "tag_b": db.get("tag", "object"),
                "pair_type": "added_in_B",
                "box_a": None,
                "box_b": db["box"],
                "crop_a": make_placeholder_image("ADDED IN B"),
                "same_location_crop_a": same_loc_crop_a,
                "crop_b": crop_from_box(img_b, db["box"]),
                "crop_b_flipped": crop_from_box(img_b, db["box"]),
                "score_a": 0.0,
                "score_b": float(db.get("score", 0.0)),
                "bbox_iou": 0.0,
                "expanded_iou": 0.0,
                "center_dist": 1.0,
                "area_ratio": 1.0,
                "match_cost": 999.0,
                "dinov2_dist": float(same_loc_evidence["normal_dino"]),
                "color_score": float(same_loc_evidence["normal_color"]),
                "flipped_dino": float(same_loc_evidence["flipped_dino"]),
                "flipped_color": float(same_loc_evidence["flipped_color"]),
                "dino_flip_improvement": 0.0,
                "color_flip_improvement": 0.0,
                "normal_edge": 0.0,
                "flipped_edge": 0.0,
                "edge_flip_improvement": 0.0,
                "change_type": "added_in_B",
                "fallback_change_type": "added_in_B",
                "fallback_importance": 5,
                "importance": 5,
                "confidence": "pre_qwen",
                "description": "",
                "change_score": 35.0,
            }

            results.append(pair)

    return results


# =========================================================
# 8. Qwen Judge / Explanation
# =========================================================

def qwen_name_single_crop(crop, side="Image B"):
    path_crop = save_temp_image(crop, f"name_single_{side.replace(' ', '_')}")

    instruction = f"""
You are given one evidence crop from {side}.

Task:
Identify the main visible foreground object in this crop.

Rules:
- Look only at this crop.
- Return a short object name.
- If the crop is mostly background, return "selected region".
- Do not compare with another image.
- Do not describe changes.
- Return ONLY valid JSON object. No markdown.

JSON format:
{{
  "object": "object name",
  "confidence": "high"
}}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path_crop},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output = qwen_processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    cleanup()

    parsed = parse_json_any(output)

    if not isinstance(parsed, dict):
        return "selected region", "low", output

    obj = str(parsed.get("object", "selected region")).strip()
    conf = str(parsed.get("confidence", "medium")).strip().lower()

    if not obj:
        obj = "selected region"

    return obj, conf, output


def qwen_judge_pair(img_a, img_b, pair):
    """
    Qwen은 change_type을 최종 결정하지 않는다.
    선택된 crop evidence를 보고 object name과 자연어 description만 생성한다.
    """
    pair_idx = pair.get("idx", "tmp")

    path_crop_a = save_temp_image(pair["crop_a"], f"explain_crop_A_{pair_idx}")
    path_crop_b = save_temp_image(pair["crop_b"], f"explain_crop_B_{pair_idx}")

    evidence = {
        "detector_tag_a": pair.get("tag_a", ""),
        "detector_tag_b": pair.get("tag_b", ""),
        "detector_display_tag": pair.get("tag", "object"),
        "detector_tag_note": "Detector tags are weak grounding hints, not final object names.",
        "system_change_type": pair.get("fallback_change_type", pair.get("change_type", "")),
        "bbox_iou": round(float(pair.get("bbox_iou", 0.0)), 3),
        "expanded_iou": round(float(pair.get("expanded_iou", 0.0)), 3),
        "center_distance": round(float(pair.get("center_dist", 0.0)), 3),
        "dinov2_distance": round(float(pair.get("dinov2_dist", 0.0)), 3),
        "color_difference": round(float(pair.get("color_score", 0.0)), 1),
        "orientation_dinov2_improvement": round(float(pair.get("dino_flip_improvement", 0.0)), 3),
    }

    change_type = pair.get("fallback_change_type", pair.get("change_type", "region_changed"))

    instruction = f"""
    You are given two evidence crop images:
    1) Evidence crop from Image A
    2) Evidence crop from Image B

    The crop pair was selected by object detection and position-based matching.
    The system-estimated change type is: {change_type}

    Evidence values:
    {json.dumps(evidence, indent=2)}

    Your task:
    Look at the two crops and describe the visible difference.

    Rules:
    - Only compare Evidence crop A and Evidence crop B.
    - Describe Image A first and Image B second.
    - Name what you actually see in the crops.
    - Detector tags are weak hints, not final object names.
    - Do not mention objects outside the crops.
    - Do not claim an object is missing or added unless the crop clearly shows that.
    - If the same main object is visible in both crops and its pose, direction, or orientation is clearly different, set orientation_change_visible to true.
    - If the main change is only color while the shape and orientation stay similar, describe the color change.
    - If the two crops show different object-like shapes, describe it as a region/object-like shape change.
    - If the exact object name is uncertain, use "selected region" or "object-like region".
    - Return ONLY valid JSON object.
    - No markdown.

    JSON format:
    {{
    "object": "object or visual region",
    "orientation_change_visible": false,
    "description": "A concrete one-sentence explanation from Image A to Image B.",
    "confidence": "high"
    }}
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path_crop_a},
                {"type": "image", "image": path_crop_b},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=260,
            do_sample=False,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output = qwen_processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    cleanup()

    parsed = parse_json_any(output)

    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        return None

    obj = str(parsed.get("object", "")).strip()
    if not obj:
        obj = pair.get("tag", "object")

    description = str(parsed.get("description", "")).strip()
    if not description:
        return None

    confidence = str(parsed.get("confidence", "medium")).lower()

    orientation_change_visible = parsed.get("orientation_change_visible", False)

    if isinstance(orientation_change_visible, str):
        orientation_change_visible = orientation_change_visible.strip().lower() in ["true", "yes", "1"]
    else:
        orientation_change_visible = bool(orientation_change_visible)
        
    return {
        "object": obj,
        "orientation_change_visible": orientation_change_visible,
        "description": description,
        "confidence": confidence,
        "raw_qwen": output,
    }


# =========================================================
# 9. Full Pipeline
# =========================================================

def qwen_description_mentions_orientation(qwen_judgement):
    if qwen_judgement is None:
        return False

    text = " ".join([
        str(qwen_judgement.get("object", "")),
        str(qwen_judgement.get("description", "")),
    ]).lower()

    orientation_words = [
        "orientation", "direction", "pose", "turned", "rotated",
        "faces", "facing", "opposite direction", "opposite orientation"
    ]

    return any(w in text for w in orientation_words)


def can_override_color_to_orientation(pair, qwen_judgement):
    if qwen_judgement is None:
        return False

    if pair.get("pair_type", "matched") != "matched":
        return False

    if pair.get("tag_a", "") != pair.get("tag_b", ""):
        return False

    current_type = pair.get("fallback_change_type", pair.get("change_type", ""))

    if current_type not in ["color_changed", "appearance_changed"]:
        return False

    expanded_iou = float(pair.get("expanded_iou", 0.0))
    center = float(pair.get("center_dist", 0.0))

    if expanded_iou < 0.20 and center > 0.12:
        return False

    return bool(qwen_judgement.get("orientation_change_visible", False))

def run_pipeline(
    img_a,
    img_b,
    max_tags=12,
    dino_box_thr=0.10,
    dino_text_thr=0.10,
    match_iou_thr=0.05,
    center_dist_thr=0.28,
    verify_threshold=0.0,
    enable_missing_added=True,
):
    img_a, img_b = ensure_same_size(img_a, img_b)

    tags, raw_tags = qwen_extract_object_tags(
        img_a,
        img_b,
        max_tags=int(max_tags),
    )

    dets_a = detect_all_tags(
        img_a,
        tags,
        box_thr=float(dino_box_thr),
        text_thr=float(dino_text_thr),
    )

    dets_b = detect_all_tags(
        img_b,
        tags,
        box_thr=float(dino_box_thr),
        text_thr=float(dino_text_thr),
    )

    pairs = match_boxes_and_score(
        img_a,
        img_b,
        dets_a,
        dets_b,
        match_iou_thr=float(match_iou_thr),
        center_dist_thr=float(center_dist_thr),
        verify_threshold=float(verify_threshold),
        enable_missing_added=bool(enable_missing_added),
    )

    judged_pairs = []

    for i, p in enumerate(pairs):
        p["idx"] = i + 1

        algorithmic_type = p.get("fallback_change_type", p.get("change_type", "region_changed"))
        algorithmic_importance = p.get("fallback_importance", p.get("importance", 3))

        if p.get("pair_type") == "added_in_B":
            obj, conf, raw = qwen_name_single_crop(p["crop_b"], side="Image B")

            qwen_judgement = {
                "object": obj,
                "description": f"The {obj} appears in Image B but not in Image A.",
                "confidence": conf,
                "raw_qwen": raw,
            }

        elif p.get("pair_type") == "missing_in_B":
            obj, conf, raw = qwen_name_single_crop(p["crop_a"], side="Image A")

            qwen_judgement = {
                "object": obj,
                "description": f"The {obj} appears in Image A but is missing in Image B.",
                "confidence": conf,
                "raw_qwen": raw,
            }

        else:
            qwen_judgement = qwen_judge_pair(img_a, img_b, p)

        final_type = algorithmic_type
        final_importance = algorithmic_importance

        if can_override_color_to_orientation(p, qwen_judgement):
            final_type = "orientation_changed"
            final_importance = max(5, int(algorithmic_importance))

        p["change_type"] = final_type
        p["importance"] = final_importance

        if qwen_judgement is not None:
            qwen_description = qwen_judgement.get("description", "")
            qwen_object = qwen_judgement.get("object", "")
            qwen_confidence = qwen_judgement.get("confidence", "qwen_explanation")
            raw_qwen = qwen_judgement.get("raw_qwen", "")

            if qwen_object:
                p["object"] = qwen_object

            if p.get("change_type") == "orientation_changed":
                obj_name = p.get("object", p.get("tag", "object"))
                p["description"] = f"The {obj_name} has a different orientation between Image A and Image B."

            elif p.get("change_type") == "object_or_region_changed":
                p["description"] = "The selected region shows a different object-like shape between Image A and Image B."

            elif qwen_description:
                p["description"] = qwen_description

            else:
                p["description"] = fallback_description_for_pair(p)

            p["confidence"] = qwen_confidence
            p["raw_qwen"] = raw_qwen
        else:
            p["confidence"] = "fallback"
            p["description"] = fallback_description_for_pair(p)

        if p.get("change_type") == "orientation_changed":
            desc = str(p.get("description", "")).lower()
            if not qwen_description_mentions_orientation({"description": desc, "object": p.get("object", "")}):
                obj_name = p.get("object", p.get("tag", "object"))
                p["description"] = f"The {obj_name} has a different orientation between Image A and Image B."

        p = veto_qwen_judgement(p)

        if p.get("change_type") == "minor_or_same":
            continue

        p["change_score"] = compute_change_score(p)
        judged_pairs.append(p)

    judged_pairs = select_diverse_final(judged_pairs)

    for i, p in enumerate(judged_pairs):
        p["idx"] = i + 1

    return tags, raw_tags, dets_a, dets_b, judged_pairs


# =========================================================
# 10. Visualization
# =========================================================
def draw_detections(image, dets):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, d in enumerate(dets):
        box = clamp_box(d.get("box"), w, h)

        if box is None:
            continue

        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{d.get('tag', '')} {float(d.get('score', 0.0)):.2f}"
        ly1 = max(0, y1 - 24)
        lx2 = min(w - 1, x1 + 260)

        draw.rectangle([x1, ly1, lx2, y1], fill=color)
        draw.text((x1 + 4, ly1 + 5), label[:38], fill=(255, 255, 255))

    return img

def draw_final_boxes(image, pairs, side="a"):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, p in enumerate(pairs):
        box = p.get(f"box_{side}")
        box = clamp_box(box, w, h)

        if box is None:
            continue

        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        label = f"{i + 1}. {p.get('change_type', '')}"
        ly1 = max(0, y1 - 30)
        lx2 = min(w - 1, x1 + 430)

        draw.rectangle([x1, ly1, lx2, y1], fill=color)
        draw.text((x1 + 5, ly1 + 6), label[:55], fill=(255, 255, 255))

    return img

def draw_wrapped_text(draw, text, xy, max_chars=48, line_height=18, fill=(40, 40, 40)):
    if text is None:
        text = ""

    text = str(text).strip()

    if not text:
        return 0

    lines = textwrap.wrap(text, width=max_chars)

    x, y = xy

    for line in lines:
        draw.text((x, y), line, fill=fill)
        y += line_height

    return len(lines) * line_height

def make_montage(pairs, crop_size=180):
    if not pairs:
        return make_placeholder_image("NO RESULTS", width=900, height=320)

    rows = len(pairs)

    row_gap = 270
    width = crop_size * 2 + 560
    height = rows * (crop_size + row_gap) + 35

    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    y = 25

    for i, p in enumerate(pairs):
        if p.get("pair_type") == "added_in_B":
            crop_a_vis = p.get("same_location_crop_a", p.get("crop_a"))
            crop_b_vis = p.get("crop_b")
        elif p.get("pair_type") == "missing_in_B":
            crop_a_vis = p.get("crop_a")
            crop_b_vis = p.get("same_location_crop_b", p.get("crop_b"))
        else:
            crop_a_vis = p.get("crop_a")
            crop_b_vis = p.get("crop_b")

        ca = crop_a_vis.resize(
            (crop_size, crop_size),
            Image.LANCZOS,
        )
        cb = crop_b_vis.resize(
            (crop_size, crop_size),
            Image.LANCZOS,
        )

        tag = p.get("tag", "object")
        qwen_object = p.get("object", "")
        change_type = p.get("change_type", "")
        pair_type = p.get("pair_type", "")
        description = p.get("description", "")
        iou = float(p.get("bbox_iou", 0.0))
        center = float(p.get("center_dist", 0.0))
        dino = float(p.get("dinov2_dist", 0.0))
        color_score = float(p.get("color_score", 0.0))
        score = float(p.get("change_score", 0.0))

        flipped_dino = float(p.get("flipped_dino", 0.0))
        dino_flip_imp = float(p.get("dino_flip_improvement", 0.0))
        flipped_color = float(p.get("flipped_color", 0.0))
        color_flip_imp = float(p.get("color_flip_improvement", 0.0))

        confidence = p.get("confidence", "")

        color = BOX_COLORS[i % len(BOX_COLORS)]

        draw.text((15, y + 10), f"P{i + 1}", fill=(0, 0, 0))
        draw.text((15, y + 35), f"detector prompt: {tag}"[:48], fill=(0, 0, 0))
        draw.text((15, y + 60), f"explained object: {qwen_object}"[:48], fill=(60, 60, 60))
        draw.text((15, y + 85), f"type: {pair_type}"[:48], fill=(60, 60, 60))
        draw.text((15, y + 110), f"change: {change_type}"[:48], fill=(60, 60, 60))
        draw.text((15, y + 135), f"conf: {confidence}"[:48], fill=(60, 60, 60))
        draw.text((15, y + 160), f"IoU: {iou:.3f} / center: {center:.3f}", fill=(60, 60, 60))
        draw.text((15, y + 185), f"DINOv2: {dino:.3f} / color: {color_score:.1f}", fill=(60, 60, 60))
        draw.text((15, y + 210), f"score: {score:.1f}", fill=(60, 60, 60))
        draw.text((15, y + 235), f"flip dino: {flipped_dino:.3f} / imp: {dino_flip_imp:.3f}", fill=(60, 60, 60))
        draw.text((15, y + 260), f"flip color: {flipped_color:.1f} / imp: {color_flip_imp:.1f}", fill=(60, 60, 60))

        x_a = 430
        x_b = 430 + crop_size + 45

        draw.text((x_a, y - 18), "Image A evidence crop", fill=(180, 0, 0))
        draw.text((x_b, y - 18), "Image B evidence crop", fill=(0, 70, 180))

        canvas.paste(ca, (x_a, y))
        canvas.paste(cb, (x_b, y))

        draw.rectangle([x_a, y, x_a + crop_size, y + crop_size], outline=color, width=3)
        draw.rectangle([x_b, y, x_b + crop_size, y + crop_size], outline=color, width=3)

        desc_y = y + crop_size + 18
        draw.text((x_a, desc_y), "Qwen explanation:", fill=(0, 0, 0))
        draw_wrapped_text(
            draw,
            description,
            (x_a, desc_y + 22),
            max_chars=78,
            line_height=18,
            fill=(40, 40, 40),
        )

        y += crop_size + row_gap

    return canvas

# =========================================================
# 11. Text Formatting
# =========================================================
def tags_to_text(tags, raw_tags):
    lines = ["[1) Qwen2-VL Object Tags]"]
    lines.append(", ".join(tags) if tags else "No tags.")
    lines.append("")
    lines.append("[Raw Qwen Output]")
    lines.append(str(raw_tags))
    return "\n".join(lines)

def detections_to_text(dets_a, dets_b):
    lines = ["[2) Grounding DINO Detections]"]

    lines.append("")
    lines.append("[Image A]")
    if not dets_a:
        lines.append("No detections.")
    else:
        for i, d in enumerate(dets_a):
            lines.append(
                f"A{i + 1}: tag={d.get('tag')} score={d.get('score', 0.0):.3f} "
                f"box={[round(v, 1) for v in d.get('box', [])]}"
            )

    lines.append("")
    lines.append("[Image B]")
    if not dets_b:
        lines.append("No detections.")
    else:
        for i, d in enumerate(dets_b):
            lines.append(
                f"B{i + 1}: tag={d.get('tag')} score={d.get('score', 0.0):.3f} "
                f"box={[round(v, 1) for v in d.get('box', [])]}"
            )

    return "\n".join(lines)

def results_to_text(pairs):
    lines = ["[3) Position-Matched Evidence + Qwen Explanation]"]

    if not pairs:
        lines.append("No verified differences.")
        return "\n".join(lines)

    for p in pairs:
        lines.append(f"[Difference {p.get('idx')}]")
        lines.append(f"- display tag: {p.get('tag')}")
        lines.append(f"- tag_a/tag_b: {p.get('tag_a')} / {p.get('tag_b')}")
        lines.append(f"- pair_type: {p.get('pair_type')}")
        lines.append(f"- change_type: {p.get('change_type')}")
        lines.append(f"- confidence: {p.get('confidence')}")
        lines.append(f"- importance: {p.get('importance')}")
        lines.append(f"- bbox_iou: {p.get('bbox_iou', 0.0):.4f}")
        lines.append(f"- expanded_iou: {p.get('expanded_iou', 0.0):.4f}")
        lines.append(f"- center_dist: {p.get('center_dist', 0.0):.4f}")
        lines.append(f"- area_ratio: {p.get('area_ratio', 0.0):.4f}")
        lines.append(f"- DINOv2 distance: {p.get('dinov2_dist', 0.0):.4f}")
        lines.append(f"- color score: {p.get('color_score', 0.0):.2f}")
        lines.append(f"- flipped DINOv2: {p.get('flipped_dino', 0.0):.4f}")
        lines.append(f"- DINOv2 flip improvement: {p.get('dino_flip_improvement', 0.0):.4f}")
        lines.append(f"- flipped color: {p.get('flipped_color', 0.0):.2f}")
        lines.append(f"- color flip improvement: {p.get('color_flip_improvement', 0.0):.2f}")
        lines.append(f"- change score: {p.get('change_score', 0.0):.2f}")
        lines.append(f"- description: {p.get('description', '')}")
        lines.append("")

    return "\n".join(lines)

# =========================================================
# 12. Gradio Callback
# =========================================================
def cb_run(
    img_a,
    img_b,
    max_tags,
    dino_box_thr,
    dino_text_thr,
    match_iou_thr,
    center_dist_thr,
    verify_threshold,
):
    try:
        if img_a is None or img_b is None:
            return (
                "",
                "",
                "",
                None,
                None,
                None,
                None,
                None,
                "두 이미지를 업로드하세요.",
            )

        img_a, img_b = ensure_same_size(img_a, img_b)

        tags, raw_tags, dets_a, dets_b, pairs = run_pipeline(
            img_a,
            img_b,
            max_tags=int(max_tags),
            dino_box_thr=float(dino_box_thr),
            dino_text_thr=float(dino_text_thr),
            match_iou_thr=float(match_iou_thr),
            center_dist_thr=float(center_dist_thr),
            verify_threshold=float(verify_threshold),
        )

        tag_text = tags_to_text(tags, raw_tags)
        det_text = detections_to_text(dets_a, dets_b)
        result_text = results_to_text(pairs)

        det_vis_a = draw_detections(img_a, dets_a)
        det_vis_b = draw_detections(img_b, dets_b)

        final_vis_a = draw_final_boxes(img_a, pairs, side="a")
        final_vis_b = draw_final_boxes(img_b, pairs, side="b")

        montage = make_montage(pairs)

        status = f"완료: tags={len(tags)}, dets_A={len(dets_a)}, dets_B={len(dets_b)}, results={len(pairs)}"

        return (
            tag_text,
            det_text,
            result_text,
            det_vis_a,
            det_vis_b,
            final_vis_a,
            final_vis_b,
            montage,
            status,
        )

    except Exception as e:
        cleanup()
        tb = traceback.format_exc()
        print(tb)

        return (
            "",
            "",
            tb,
            None,
            None,
            None,
            None,
            None,
            f"오류: {str(e)}",
        )

# =========================================================
# 13. UI
# =========================================================
with gr.Blocks(title="Evidence-Grounded Visual Difference Explainer") as demo:
    gr.Markdown(
        """

This demo uses **three foundation models**:

1. **Qwen2-VL** proposes object tags from both images.
2. **Grounding DINO** grounds those tags into detected boxes.
3. **DINOv2** compares position-matched evidence crops using visual feature distance.

Qwen2-VL verbalizes the selected crop evidence into a natural-language explanation.
        """
    )

    with gr.Row():
        image_a = gr.Image(type="pil", label="Image A")
        image_b = gr.Image(type="pil", label="Image B")

    with gr.Row():
        max_tags = gr.Slider(
            3,
            20,
            value=MAX_TAGS_DEFAULT,
            step=1,
            label="Max Qwen object tags",
        )
        dino_box_thr = gr.Slider(
            0.05,
            0.80,
            value=DINO_BOX_THR_DEFAULT,
            step=0.01,
            label="Grounding DINO box threshold",
        )
        dino_text_thr = gr.Slider(
            0.05,
            0.80,
            value=DINO_TEXT_THR_DEFAULT,
            step=0.01,
            label="Grounding DINO text threshold",
        )

    with gr.Row():
        match_iou_thr = gr.Slider(
            0.00,
            0.80,
            value=MATCH_IOU_THR_DEFAULT,
            step=0.01,
            label="Box match IoU threshold",
        )
        center_dist_thr = gr.Slider(
            0.05,
            0.80,
            value=CENTER_DIST_THR_DEFAULT,
            step=0.01,
            label="Box match center-distance threshold",
        )
        verify_threshold = gr.Slider(
            0.00,
            50.0,
            value=VERIFY_THRESHOLD_DEFAULT,
            step=0.5,
            label="Change score threshold",
        )

    run_btn = gr.Button("Run Position-Based Box Matching Pipeline", variant="primary")

    status = gr.Textbox(label="Status", lines=2)

    tag_text = gr.Textbox(label="1) Qwen2-VL Object Tags", lines=10)
    det_text = gr.Textbox(label="2) Grounding DINO Detections", lines=18)
    result_text = gr.Textbox(label="3) Position-Matched Differences", lines=22)

    with gr.Row():
        det_vis_a = gr.Image(type="pil", label="All Grounding DINO Boxes - Image A")
        det_vis_b = gr.Image(type="pil", label="All Grounding DINO Boxes - Image B")

    with gr.Row():
        final_vis_a = gr.Image(type="pil", label="Final Difference Boxes - Image A")
        final_vis_b = gr.Image(type="pil", label="Final Difference Boxes - Image B")
    
    montage = gr.Image(type="pil", label="Evidence Crop Montage")

    run_btn.click(
        fn=cb_run,
        inputs=[
            image_a,
            image_b,
            max_tags,
            dino_box_thr,
            dino_text_thr,
            match_iou_thr,
            center_dist_thr, 
            verify_threshold,
        ],
        outputs=[
            tag_text,
            det_text,
            result_text,
            det_vis_a,
            det_vis_b,
            final_vis_a,
            final_vis_b,
            montage,
            status,
        ],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0")
