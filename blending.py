"""
В этом файле находится этап 4 пайплайна - blending.

Берёт результат инпейнтинга (этап 3) и бесшовно сшивает перерисованную
область с оригинальной иллюстрацией методом feathering

Как работает:
1) Загружает оригинал, результат инпейнтинга и маску этапа 2.
2) Сшивает feathering'ом.
3) Результат сохраняется в виде:
   blended_feather.png   - итог
   blend_compare.jpg     - для сравнения и подбора alpha

Запуск:
  python blending.py --generated data/outputs/spread_01__ivan/best.png

!!! Требует результат инпейнтинга (output_inpainting.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from illustrations_mask import PROJECT_ROOT, ILLUSTRATIONS_DIR, MASKS_DIR
from illustrations_detect import IMAGE_EXTS


DEFAULT_FEATHER = 8            # радиус размытия края маски, px
CROP_MARGIN = 0.35             # запас вокруг маски для визуализации


# ----------------------------- поиск входов по имени --------------------------

def resolve_inputs(generated_path: Path,
                   original_arg: str | None,
                   mask_arg: str | None) -> tuple[Path, Path]:

    if original_arg and mask_arg:
        return Path(original_arg), Path(mask_arg)

    folder = generated_path.parent.name
    if "__" not in folder:
        raise ValueError(
            f"Не удалось определить иллюстрацию по папке '{folder}'. "
        )
    stem = folder.rsplit("__", 1)[0]

    original = Path(original_arg) if original_arg else None
    if original is None:
        for ext in IMAGE_EXTS:
            cand = ILLUSTRATIONS_DIR / f"{stem}{ext}"
            if cand.exists():
                original = cand
                break
    if original is None:
        raise FileNotFoundError(
            f"Иллюстрация '{stem}.*' не найдена в {ILLUSTRATIONS_DIR}. "
        )

    mask = Path(mask_arg) if mask_arg else MASKS_DIR / f"{stem}_mask.png"
    return original, mask


# ----------------------------- geometry ---------------------------------------

def mask_crop_box(mask: np.ndarray, margin: float):
    """Прямоугольник вокруг маски с запасом (для визуализации). """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Маска пустая.")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = mask.shape
    mx = int(round((x2 - x1) * margin))
    my = int(round((y2 - y1) * margin))
    return (max(0, x1 - mx), max(0, y1 - my),
            min(w - 1, x2 + mx), min(h - 1, y2 + my))


# ----------------------------- blending ---------------------------------------

def blend_feather(original: np.ndarray, generated: np.ndarray,
                  mask: np.ndarray, feather: int) -> np.ndarray:
    """
    Альфа-смешивание по размытому краю маски.
    alpha=1 внутри маски, плавно спадает к 0 у края -> мягкий шов.
    """
    if feather <= 0:
        out = original.copy()
        out[mask > 127] = generated[mask > 127]
        return out
    k = feather * 2 + 1
    alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    alpha = alpha[..., None]  # HxWx1
    blended = generated.astype(np.float32) * alpha + \
        original.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ----------------------------- визуализация -----------------------------------

def make_compare(generated: np.ndarray, blended: np.ndarray,
                 mask: np.ndarray, feather: int, out_path: Path) -> None:
    """Зум на зону маски: hard paste (вход) и feather рядом, с подписями."""
    x1, y1, x2, y2 = mask_crop_box(mask, CROP_MARGIN)

    def tile(img: np.ndarray, label: str) -> np.ndarray:
        t = img[y1:y2, x1:x2]
        scale = 512.0 / max(t.shape[:2])
        t = cv2.resize(t, (int(t.shape[1] * scale), int(t.shape[0] * scale)))
        canvas = np.full((560, t.shape[1], 3), 25, np.uint8)
        canvas[:t.shape[0]] = t
        cv2.putText(canvas, label, (10, 545), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        return canvas

    compare = np.hstack([
        tile(generated, "hard paste (input)"),
        tile(blended, f"feather (r={feather})"),
    ])
    cv2.imwrite(str(out_path), compare)


# ----------------------------------- main -------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blending (feather)"
    )
    parser.add_argument("--generated", required=True,
                        help="Результат этапа 3 (best.png).")
    parser.add_argument("--original", default=None,
                        help="Оригинальная иллюстрация. По умолчанию ищется по имени папки.")
    parser.add_argument("--mask", default=None,
                        help="Маска этапа 2. По умолчанию ищется по имени ппаки.")
    parser.add_argument("--feather", type=int, default=DEFAULT_FEATHER,
                        help="Радиус размытия края маски, px.")
    args = parser.parse_args()

    generated_path = Path(args.generated)
    if not generated_path.exists():
        raise FileNotFoundError(f"Не найден результат инпейнтинга: {generated_path}")
    original_path, mask_path = resolve_inputs(generated_path, args.original, args.mask)
    for p, what in [(original_path, "иллюстрация"), (mask_path, "маска")]:
        if not p.exists():
            raise FileNotFoundError(f"Не найден файл ({what}): {p}")

    original = cv2.imread(str(original_path))
    generated = cv2.imread(str(generated_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if original is None or generated is None or mask is None:
        raise ValueError("Не удалось прочитать один из входных файлов.")
    if not (original.shape[:2] == generated.shape[:2] == mask.shape[:2]):
        raise ValueError(
            f"Размеры не совпадают: original={original.shape[:2]}, "
            f"generated={generated.shape[:2]}, mask={mask.shape[:2]}"
        )

    print(f"Оригинал:  {original_path.name}")
    print(f"Генерация: {generated_path}")
    print(f"Маска:     {mask_path.name}")
    print(f"Feather:   r={args.feather}")

    out_dir = generated_path.parent

    blended = blend_feather(original, generated, mask, args.feather)
    blended_path = out_dir / "blended_feather.png"
    cv2.imwrite(str(blended_path), blended)
    print(f"Итог: {blended_path}")

    compare_path = out_dir / "blend_compare.jpg"
    make_compare(generated, blended, mask, args.feather, compare_path)
    print(f"Сравнение (зум на шов): {compare_path}")


if __name__ == "__main__":
    main()