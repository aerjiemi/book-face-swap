"""
В этом файле находится этап 2 пайплайна.

С помощью SAM2 и JSON из RetinaFace, сделанного на этапе 1, строится маска лицо+волосы

Как работает:
1) bbox расширяется, чтобы он захватывал волосы + лицо
2) добавляются якоря (landmarks + positive-точки в зоне волос)
3) SAM2 делает маску
4) Результат сохраняется в виде:
бинарная  (data/masks/<name>_mask.png)
визуализация для проверки (data/masks/<name>_vis.jpg)
JSON с промптами/score  (data/masks/<name>.json)


Маска дальше пойдёт в stylized inpainting как область, которую модель
перерисовывает под лицо+причёску клиента.

Запуск:
  python illustrations_mask.py                       # выбрать из списка вручную
  python illustrations_mask.py --image data/illustrations/scene1.png

!!! Требует, чтобы для картинки уже был посчитан детект (illustrations_detect.py).
"""


from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from illustrations_detect import (
    list_illustrations, choose_illustration, PROJECT_ROOT,
    ILLUSTRATIONS_DIR, DETECTIONS_DIR, IMAGE_EXTS)


MASKS_DIR = PROJECT_ROOT / "data" / "masks"

DEFAULT_MODEL = "facebook/sam2.1-hiera-large"

# Расширение face-bbox под волосы (доли от размера лица). Подбирается.
DEFAULT_PAD_UP = 0.8     # вверх (волосы надо лбом)
DEFAULT_PAD_DOWN = 0.15  # вниз (подбородок / немного шеи)
DEFAULT_PAD_SIDE = 0.25  # влево-вправо (волосы по бокам лица)

# Сколько точек-якорей ставить в зоне волос (0 = выключить). Подбирается.
DEFAULT_HAIR_POINTS = 3

# Постобработка маски (чинит дырки от SAM2 с помощью заливки).
DEFAULT_CLOSE_KERNEL = 5    # ядро morphological close (0 = выкл)
DEFAULT_MIN_COMP_FRAC = 0.05  # мин. доля площади от крупнейшей компоненты


def load_detection(stem: str) -> dict:
    """Читает JSON, сохранённый на этапе 1."""
    json_path = DETECTIONS_DIR / f"{stem}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Нет детекта для '{stem}': {json_path}\n"
            f"Сначала необходимо запустить illustrations_detect.py для этой картинки."
        )
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("detections"):
        raise ValueError(f"В {json_path} нет детекций.")
    return data


def expand_box(bbox, img_w, img_h, pad_up, pad_down, pad_side) -> list[int]:
    """Расширяет face-bbox, чтобы box-промпт накрыл волосы. Клампит по границам."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    nx1 = max(0, int(round(x1 - pad_side * w)))
    ny1 = max(0, int(round(y1 - pad_up * h)))
    nx2 = min(img_w - 1, int(round(x2 + pad_side * w)))
    ny2 = min(img_h - 1, int(round(y2 + pad_down * h)))
    return [nx1, ny1, nx2, ny2]


def hair_points(face_bbox, prompt_box, n) -> list[list[float]]:
    """
        n - Positive-точки-якоря в зоне волос.
        n=1 -> одна по центру
        n=2 -> две по бокам
        n>=3 -> три (по бокам + центр)
    """
    if n <= 0:
        return []
    x1, _, x2, _ = face_bbox
    w = x2 - x1
    cx = (x1 + x2) / 2.0
    face_top = face_bbox[1]
    box_top = prompt_box[1]
    hy = (face_top + box_top) / 2.0

    if n == 1:
        xs = [cx]
    elif n == 2:
        xs = [x1 + 0.30 * w, x2 - 0.30 * w]
    else:
        xs = [x1 + 0.25 * w, cx, x2 - 0.25 * w]
    return [[float(x), float(hy)] for x in xs]


def build_predictor(model_id: str, device: str) -> SAM2ImagePredictor:
    print(f"Загрузка SAM2: {model_id} (device={device}) ")
    return SAM2ImagePredictor.from_pretrained(model_id, device=device)


def predict_mask(predictor, box, points, labels, device):
    """ Создание маски с помощью SAM2. Берём с max score."""
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda" else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            box=np.array(box, dtype=np.float32),
            multimask_output=True,
        )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Заливает полностью замкнутые дыры внутри маски """
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    seed = None
    for c in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if mask_u8[c[1], c[0]] == 0:
            seed = c
            break
    if seed is None:
        return mask_u8
    cv2.floodFill(flood, ff, seed, 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, holes)


def keep_components(mask_u8: np.ndarray, frac: float) -> np.ndarray:
    """Убирает мелкие отдельные кляксы (например, от якоря, попавшего в фон)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    amax = int(areas.max())
    out = np.zeros_like(mask_u8)
    for i, a in enumerate(areas, start=1):
        if a >= frac * amax:
            out[labels == i] = 255
    return out


def clean_mask(mask_bool: np.ndarray, close_kernel: int, min_comp_frac: float) -> np.ndarray:
    """
    Постобработка маски SAM2:
      1) close   — сшивает узкие разрывы по краю;
      2) fill_holes    — заливает замкнутые дыры ;
      3) keep_components  — удаляет мелкие кляксы.
    Возвращает bool-маску.
    """
    u8 = mask_bool.astype(np.uint8) * 255
    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
    u8 = fill_holes(u8)
    if min_comp_frac > 0:
        u8 = keep_components(u8, min_comp_frac)
    return u8 > 0


def draw_visualization(image_bgr, combined_mask, boxes, points_by_face, out_path):
    """ bbox + маска визуализация """
    vis = image_bgr.copy()

    overlay = vis.copy()
    overlay[combined_mask] = (0, 200, 0)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    contours, _ = cv2.findContours(
        combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

    for pf in points_by_face:
        for (px, py) in pf["facial"]:          # лицо — красные
            cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)
        for (px, py) in pf["hair"]:            # волосы — голубые
            cv2.circle(vis, (int(px), int(py)), 4, (255, 200, 0), -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Маска лица+волос на иллюстрации с помощью SAM2."
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Путь к иллюстрации. Если не задан — выбор из списка.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="ID модели SAM2 на Hugging Face.")
    parser.add_argument("--pad-up", type=float, default=DEFAULT_PAD_UP,
                        help="Расширение box вверх (волосы)")
    parser.add_argument("--pad-down", type=float, default=DEFAULT_PAD_DOWN,
                        help="Расширение box вниз ")
    parser.add_argument("--pad-side", type=float, default=DEFAULT_PAD_SIDE,
                        help="Расширение box по бокам")
    parser.add_argument("--hair-points", type=int, default=DEFAULT_HAIR_POINTS,
                        help="Кол-во positive-points в зоне волос (0=выкл).")
    parser.add_argument("--close-kernel", type=int, default=DEFAULT_CLOSE_KERNEL,
                        help="Ядро morphological close (0=выкл). Сшивает дыры в маске.")
    parser.add_argument("--min-comp-frac", type=float, default=DEFAULT_MIN_COMP_FRAC,
                        help="Мин. доля площади от крупнейшей компоненты (убираются лишние точки на маске).")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else choose_illustration()
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    detection = load_detection(image_path.stem)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Не удалось прочитать: {image_path}")
    img_h, img_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_predictor(args.model, device)
    predictor.set_image(image_rgb)

    combined = np.zeros((img_h, img_w), dtype=bool)
    expanded_boxes = []
    points_by_face = []
    per_face_meta = []

    print(f"\nВыбрано изображение: {image_path.name} ")
    for i, det in enumerate(detection["detections"]):
        box = expand_box(det["bbox"], img_w, img_h,
                         args.pad_up, args.pad_down, args.pad_side)

        facial_pts = [pt for pt in det["landmarks"].values()]  # positive, на лице
        hair_pts = hair_points(det["bbox"], box, args.hair_points)  # positive, в волосах

        points = facial_pts + hair_pts
        labels = [1] * len(points)

        mask, score = predict_mask(predictor, box, points, labels, device)
        mask = clean_mask(mask, args.close_kernel, args.min_comp_frac)
        combined |= mask
        expanded_boxes.append(box)
        points_by_face.append({"facial": facial_pts, "hair": hair_pts})
        per_face_meta.append({
            "face_index": i,
            "face_bbox": det["bbox"],
            "prompt_box": box,
            "hair_points": hair_pts,
            "sam2_score": score,
        })
        print(f"  [{i}] sam2_score={score:.3f}  prompt_box={box}  hair_pts={len(hair_pts)}")

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    mask_path = MASKS_DIR / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), (combined.astype(np.uint8) * 255))

    vis_path = MASKS_DIR / f"{stem}_vis.jpg"
    draw_visualization(image_bgr, combined, expanded_boxes, points_by_face, vis_path)

    json_path = MASKS_DIR / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image": str(image_path),
            "model": args.model,
            "pad": {"up": args.pad_up, "down": args.pad_down, "side": args.pad_side},
            "hair_points": args.hair_points,
            "faces": per_face_meta,
            "mask_path": str(mask_path),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nМаска:        {mask_path}")
    print(f"Визуализация: {vis_path}")
    print(f"JSON:         {json_path}")


if __name__ == "__main__":
    main()