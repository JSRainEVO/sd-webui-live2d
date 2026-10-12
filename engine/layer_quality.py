"""Conservative source checks and visible-mouth recovery for generated PSDs.

This runs in the image engine environment. It does not generate unseen mouth
interiors. Original AI output is retained, and every applied change is recorded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from psd_tools import PSDImage

OPTIONAL_PARTS = {"wings", "tail", "objects", "headwear", "earwear", "neckwear"}
HEAD_ORDER = {"ears": 0, "earwear": 1, "face": 2, "eyewhite": 3, "irides": 4,
              "eyelash": 5, "eyebrow": 6, "nose": 7, "mouth": 8,
              "mouth close": 8, "mouth open": 8, "eyewear": 9,
              "front hair": 10, "headwear": 11}


def part_name(name: str) -> str:
    return name[:-2] if name.endswith(("-l", "-r")) else name


def normalized_source(source: Path, side: int) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGBA")
    square = Image.new("RGBA", (max(image.size),) * 2)
    square.alpha_composite(image, ((square.width - image.width) // 2,
                                   (square.height - image.height) // 2))
    return square.resize((side, side), Image.Resampling.LANCZOS)


def foreground_mask(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Trust alpha, or a uniform corner background connected to the border."""
    alpha = image[..., 3]
    valid = alpha > 240
    if np.count_nonzero(alpha < 10) > alpha.size * .05:
        # Padded opaque inputs also contain transparency. Check their corners
        # inside the valid rectangle before interpreting alpha as a silhouette.
        ys, xs = np.nonzero(valid)
        if not len(xs):
            return None, "empty_source"
        box = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if np.count_nonzero(box[..., 3] < 10) > box.shape[0] * box.shape[1] * .05:
            return alpha > 15, "source_alpha"
    else:
        box = image
    n = max(2, min(box.shape[:2]) // 30)
    corners = [box[:n, :n], box[:n, -n:], box[-n:, :n], box[-n:, -n:]]
    samples = [part[..., :3][part[..., 3] > 240] for part in corners]
    if any(len(sample) < n * n // 2 for sample in samples):
        return alpha > 15, "source_alpha"
    colors = np.array([np.median(sample, axis=0) for sample in samples])
    color = np.median(colors, axis=0)
    if np.max(np.abs(colors - color)) > 14 or any(
            np.quantile(np.max(np.abs(sample - color), axis=1), .9) > 20 for sample in samples):
        return None, "complex_background_checks_skipped"
    near = (np.max(np.abs(image[..., :3].astype(float) - color), axis=2) < 28) | (alpha < 15)
    count, labels = cv2.connectedComponents(near.astype(np.uint8), connectivity=8)
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge_ids = edge_ids[edge_ids != 0]
    background = np.isin(labels, edge_ids) if count > 1 else near
    return (~background) & (alpha > 15), "uniform_background"


def unsupported_part(layer: np.ndarray, source: np.ndarray,
                     foreground: np.ndarray,
                     occluded: np.ndarray | None = None) -> tuple[bool, dict]:
    mask = layer[..., 3] > 32
    if occluded is not None:
        mask &= ~occluded
    area = int(mask.sum())
    if not area:
        return False, {"pixels": 0}
    overlap = float(np.count_nonzero(mask & foreground) / area)
    similar = np.max(np.abs(layer[..., :3].astype(np.int16) - source[..., :3]), axis=2) < 50
    support = float(np.count_nonzero(mask & foreground & similar) / area)
    return overlap < .15 and support < .04, {
        "pixels": area, "foregroundOverlap": round(overlap, 4),
        "sourceColorSupport": round(support, 4),
    }


def recover_mouth(source: np.ndarray, face: np.ndarray,
                  eyes: list[np.ndarray]) -> tuple[np.ndarray | None, dict]:
    ys, xs = np.nonzero(face[..., 3] > 128)
    if not len(xs) or not eyes:
        return None, {"status": "skipped", "reason": "face_or_eyes_missing"}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    fw, fh = x1 - x0 + 1, y1 - y0 + 1
    eye_mask = np.logical_or.reduce([eye[..., 3] > 32 for eye in eyes])
    ey, ex = np.nonzero(eye_mask)
    if not len(ex):
        return None, {"status": "skipped", "reason": "eyes_empty"}
    centers = []
    for eye in eyes:
        ey_local, ex_local = np.nonzero(eye[..., 3] > 32)
        if len(ex_local):
            centers.append((np.median(ex_local), np.median(ey_local)))
    cx, eye_y = np.mean(centers, axis=0)
    gap = y1 - eye_y
    if gap < fh * .15:
        return None, {"status": "skipped", "reason": "face_geometry_uncertain"}
    rx0, rx1 = max(0, int(cx - fw * .22)), min(source.shape[1], int(cx + fw * .22) + 1)
    ry0, ry1 = max(0, int(eye_y + gap * .47)), min(source.shape[0], int(eye_y + gap * .84) + 1)
    crop = source[ry0:ry1, rx0:rx1]
    if min(crop.shape[:2]) < 3:
        return None, {"status": "skipped", "reason": "mouth_region_too_small"}
    gray = cv2.cvtColor(source[..., :3], cv2.COLOR_RGB2GRAY)
    kernel = max(5, int(fw * .12) | 1)
    background = cv2.medianBlur(gray, kernel)
    contrast = background.astype(np.float32) - gray.astype(np.float32)
    roi_contrast = contrast[ry0:ry1, rx0:rx1]
    candidates = (roi_contrast > 8).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    choices = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 2 or w > fw * .35 or h > fh * .14 or x == 0 or x + w == crop.shape[1]:
            continue
        if abs(centers[i, 0] + rx0 - cx) > fw * .13:
            continue
        expected_y = eye_y + gap * .65
        score = abs(centers[i, 0] + rx0 - cx) / fw + abs(centers[i, 1] + ry0 - expected_y) / gap
        choices.append((score, i))
    if not choices:
        return None, {"status": "skipped", "reason": "no_confident_visible_mouth"}
    _, chosen = min(choices)
    selected = (labels == chosen).astype(np.uint8)
    selected = cv2.dilate(selected, np.ones((3, 3), np.uint8))
    alpha = np.clip(roi_contrast * 12, 0, 255).astype(np.uint8) * selected
    mouth = np.zeros_like(source)
    mouth[ry0:ry1, rx0:rx1, :3] = crop[..., :3]
    mouth[ry0:ry1, rx0:rx1, 3] = alpha
    return mouth, {"status": "recovered_visible_line", "pixels": int((alpha > 15).sum()),
                   "region": [rx0, ry0, rx1, ry1], "mouthInterior": "not_generated"}


def refine(source_path: Path, raw_psd: Path, output: Path,
           canvas_size: tuple[int, int] | None = None) -> dict:
    raw = PSDImage.open(raw_psd)
    if raw.width != raw.height:
        raise ValueError("自动拆层 PSD 应为正方形画布")
    # Preserve tiny source details at the source's resolution. Upscaling other
    # AI layers does not restore detail; the audit records both resolutions.
    if canvas_size is not None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in canvas_size):
            raise ValueError("画布宽度和高度必须是正整数")
        with Image.open(source_path) as original:
            if ImageOps.exif_transpose(original).size != canvas_size:
                raise ValueError("输入图片必须先适配到所选画布尺寸")
        side = max(canvas_size)
    else:
        with Image.open(source_path) as original:
            side = max(raw.width, min(2048, max(original.size)))
    width, height = canvas_size or (side, side)
    left, top = (side - width) // 2, (side - height) // 2
    canvas_box = (left, top, left + width, top + height)
    src = np.asarray(normalized_source(source_path, side))
    fg, mask_method = foreground_mask(src)
    prepared = []
    for layer in raw:
        image = Image.new("RGBA", raw.size)
        image.alpha_composite(layer.composite(), (layer.left, layer.top))
        array = np.asarray(image.resize((side, side), Image.Resampling.LANCZOS))
        prepared.append((layer.name, array))
    # Support must be measured on visible pixels: hidden wings can overlap
    # source clothing without providing any evidence that wings exist.
    occlusion = [None] * len(prepared)
    covered = np.zeros((side, side), dtype=bool)
    for i in range(len(prepared) - 1, -1, -1):
        occlusion[i] = covered.copy()
        covered |= prepared[i][1][..., 3] > 200
    layers = []
    removed, checked = [], []
    clipped_pixels = 0
    envelope = None
    if fg is not None:
        radius = max(2, side // 170)
        envelope = cv2.dilate(fg.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)) > 0
    for i, (name, array) in enumerate(prepared):
        if name in OPTIONAL_PARTS and fg is not None:
            reject, metrics = unsupported_part(array, src, fg, occlusion[i])
            checked.append({"name": name, **metrics})
            if reject:
                removed.append({"name": name, "reason": "insufficient_source_support", **metrics})
                continue
        if envelope is not None:
            clipped_pixels += int(np.count_nonzero((array[..., 3] > 15) & ~envelope))
            array = array.copy()
            array[..., 3][~envelope] = 0
        layers.append((name, array))
    mouth_info = {"status": "already_present"}
    mouth_layers = [arr for name, arr in layers if name in {"mouth", "mouth close", "mouth open"}]
    face = next((arr for name, arr in layers if name == "face"), None)
    eyes = [arr for name, arr in layers if name.startswith("irides")]
    if not mouth_layers and face is not None:
        mouth, mouth_info = recover_mouth(src, face, eyes)
        if mouth is not None:
            face_index = next(i for i, (name, _) in enumerate(layers) if name == "face")
            layers.insert(face_index + 1, ("mouth close", mouth))
    # A single median depth can put spectacles behind the face or lashes
    # behind irises. For the front-facing profile, use semantic head order.
    head_layers = [(name, arr) for name, arr in layers if part_name(name) in HEAD_ORDER]
    body_layers = [(name, arr) for name, arr in layers if part_name(name) not in HEAD_ORDER]
    head_layers.sort(key=lambda pair: HEAD_ORDER[part_name(pair[0])])
    layers = body_layers + head_layers
    result = PSDImage.new("RGBA", (width, height))
    for name, array in layers:
        # Undo the inference engine's square padding in every layer, keeping
        # their positions aligned with the user's rectangular output canvas.
        image = Image.fromarray(array).crop(canvas_box)
        bbox = image.getbbox()
        if bbox:
            result.create_pixel_layer(image.crop(bbox), name=name, top=bbox[1], left=bbox[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    before = raw.composite(force=True)
    if canvas_size is not None:
        before = before.resize((side, side), Image.Resampling.LANCZOS).crop(canvas_box)
    before.save(output.with_suffix(".before.png"))
    result.composite(force=True).save(output.with_suffix(".preview.png"))
    audit = {"rawPsd": str(raw_psd), "cleanedPsd": str(output),
             "inferenceResolution": raw.width,
             "outputCanvas": {"width": width, "height": height} if canvas_size else side,
             "outputWidth": width, "outputHeight": height,
             "sourceSquareOffset": [left, top],
             "foregroundMethod": mask_method, "optionalPartChecks": checked,
             "removedLayers": removed, "mouthRecovery": mouth_info,
             "silhouetteClippedPixels": clipped_pixels,
             "drawingOrder": [name for name, _ in layers],
             "preview": str(output.with_suffix(".preview.png"))}
    output.with_suffix(".audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("部件检查：" + ("、".join(p["name"] for p in removed) + " 缺少原图依据，已移除。" if removed else "未移除部件。"))
    print("嘴部检查：" + str(mouth_info["status"]))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查自动拆层并恢复可见嘴部")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-psd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("请同时填写 --width 和 --height")
    refine(args.source, args.raw_psd, args.output,
           (args.width, args.height) if args.width is not None else None)
