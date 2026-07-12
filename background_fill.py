"""
Этап 3 пайплайна (финализация): убираем "нимб" вокруг новой головы.

Проблема: предрасчитанная маска головы (illustrations_mask) повторяет контур
ИСХОДНОГО персонажа. Если голова клиента получилась меньше / другой формы,
внутри старого контура остаётся кольцо перегенерированного псевдо-фона
("нимб"): в коллаже это место было затёрто cv2.inpaint (Telea), ControlNet
Tile честно воспроизводит эту кашу, а paste-маска (= старый контур) вклеивает
её в оригинал. Регулировкой seam_dilate/paste_dilate это НЕ лечится:
кольцо лежит ВНУТРИ paste-маски, а не снаружи.

Решение (после генерации, на готовом кадре):
  1) face-parsing (тот же SegFormer, что в hair_collage) сегментирует
     НОВУЮ голову на результате. В защиту входят: кожа лица, волосы, уши,
     глаза/брови/губы, головной убор и — критично — ШЕЯ (классы neck,
     neck_l) и ОДЕЖДА (cloth). Эти пиксели не меняются ни при каких условиях.
  2) нимб = paste-регион МИНУС защищённая зона (с запасом на края волос).
  3) нимб заливается LaMa (big-lama) — специализированная модель
     заполнения фона: продолжает текстуру/мазки окружения без промпта,
     без диффузии и без риска дорисовать "лишнее".

Гарантии безопасности:
  - лицо/волосы/шея/одежда не изменяются вообще (жёсткая защитная маска);
  - вне paste-региона кадр не трогается (там и так оригинальные пиксели);
  - если парсинг новой головы неубедителен (<30% региона) — заливка
    пропускается и кадр возвращается как есть (безопасный no-op),
    чтобы случайно не стереть голову.

Зависимость: pip install simple-lama-inpainting
(первый запуск скачает веса big-lama, ~200MB).

Standalone-отладка на уже сгенерированном кандидате:
  python background_fill.py \
      --image  data/outputs/spread_01__ivan/candidate_42.png \
      --region data/outputs/spread_01__ivan/paste_mask.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hair_collage import build_face_parser, _parse_labels

# всё, что принадлежит новому персонажу и НЕ должно затираться фоном
PROTECT_LABELS = {
    "skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "eye_g",
    "u_lip", "l_lip", "mouth", "l_ear", "r_ear", "ear_r",
    "hair", "hat",
    "neck", "neck_l",   # ШЕЯ — главное требование: не терять
    "cloth",            # воротник/плечи, если попали в регион
}

HALO_PAD_FRAC = 0.25      # запас кропа вокруг региона для парсинга
MIN_PROTECT_FRAC = 0.30   # ниже — считаем, что парсинг провалился (no-op)
MIN_HALO_FRAC = 0.01      # нимб меньше 1% региона — заливать нечего


def build_lama(device: str):
    """LaMa (big-lama) через simple-lama-inpainting."""
    import torch
    from simple_lama_inpainting import SimpleLama
    return SimpleLama(device=torch.device(device))


def _protect_mask(image_bgr: np.ndarray, region_mask: np.ndarray,
                  parser) -> np.ndarray:
    """Маска НОВОЙ головы (лицо+волосы+уши+шея+одежда) на готовом кадре.

    Парсим кроп вокруг региона (zoom повышает надёжность SegFormer),
    результат мапим обратно в полный кадр.
    """
    h, w = image_bgr.shape[:2]
    ys, xs = np.where(region_mask > 0)
    if len(xs) == 0:
        return np.zeros((h, w), np.uint8)

    pad = int(HALO_PAD_FRAC * max(xs.max() - xs.min(), ys.max() - ys.min()))
    x1 = max(0, int(xs.min()) - pad); y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad); y2 = min(h, int(ys.max()) + pad)

    labels_map = _parse_labels(image_bgr[y1:y2, x1:x2], parser)
    id2label = parser[2]
    wanted_ids = [i for i, name in id2label.items() if name in PROTECT_LABELS]
    crop_protect = np.isin(labels_map, wanted_ids).astype(np.uint8) * 255

    # закрываем мелкие дыры (блики на волосах/коже)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    crop_protect = cv2.morphologyEx(crop_protect, cv2.MORPH_CLOSE, kernel)

    protect = np.zeros((h, w), np.uint8)
    protect[y1:y2, x1:x2] = crop_protect
    return protect


class BackgroundFiller:
    """Заливка нимба. Грузит SegFormer + LaMa один раз, дальше — по кадру."""

    def __init__(self, device: str):
        self.device = device
        self.parser = build_face_parser(device)
        self.lama = build_lama(device)

    def __call__(self, image_bgr: np.ndarray, region_mask: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, str]:
        """Возвращает (кадр_после_заливки, маска_нимба, статус).

        Статусы: "lama" | "skipped_parse_failed" | "skipped_no_halo".
        При любом skip кадр возвращается без изменений.
        """
        h, w = image_bgr.shape[:2]
        region = ((region_mask > 0).astype(np.uint8)) * 255
        region_area = max(int((region > 0).sum()), 1)

        protect = _protect_mask(image_bgr, region, self.parser)

        # sanity: если внутри региона защищено слишком мало — парсинг
        # скорее всего не увидел голову; заливать опасно (сотрём персонажа)
        protect_in_region = cv2.bitwise_and(protect, region)
        if int((protect_in_region > 0).sum()) < MIN_PROTECT_FRAC * region_area:
            print("[background_fill] парсинг новой головы неубедителен — "
                  "заливка нимба пропущена (кадр не изменён).")
            return image_bgr, np.zeros((h, w), np.uint8), "skipped_parse_failed"

        # защиту слегка расширяем, чтобы не съесть тонкие пряди и край шеи
        grow = max(3, int(0.015 * np.sqrt(region_area)))
        kg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * grow + 1, 2 * grow + 1))
        protect_grown = cv2.dilate(protect, kg)

        halo = cv2.subtract(region, protect_grown)
        # убираем однопиксельный мусор по границе
        halo = cv2.morphologyEx(
            halo, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if int((halo > 0).sum()) < MIN_HALO_FRAC * region_area:
            return image_bgr, halo, "skipped_no_halo"

        # маску ГЕНЕРАЦИИ LaMa чуть расширяем наружу (в настоящий фон),
        # чтобы заливка перекрыла шов старого контура; в голову — нельзя
        halo_gen = cv2.dilate(halo, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (9, 9)))
        halo_gen = cv2.subtract(halo_gen, protect)

        # кроп вокруг нимба (LaMa быстрее и качественнее на локальном кропе)
        ys, xs = np.where(halo_gen > 0)
        pad = int(0.30 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 16
        cx1 = max(0, int(xs.min()) - pad); cy1 = max(0, int(ys.min()) - pad)
        cx2 = min(w, int(xs.max()) + pad); cy2 = min(h, int(ys.max()) + pad)

        from PIL import Image
        crop = image_bgr[cy1:cy2, cx1:cx2]
        crop_mask = halo_gen[cy1:cy2, cx1:cx2]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(crop_mask)

        result = self.lama(pil_img, pil_mask)
        filled_crop = cv2.cvtColor(np.array(result.convert("RGB")),
                                   cv2.COLOR_RGB2BGR)
        # simple-lama паддит вход до кратности 8 — обрезаем обратно
        filled_crop = filled_crop[:crop.shape[0], :crop.shape[1]]

        # мягкая вклейка СТРОГО по маске нимба; на защищённой зоне alpha=0,
        # поэтому лицо/волосы/шея/одежда физически не могут измениться
        alpha = cv2.GaussianBlur(crop_mask, (7, 7), 0).astype(np.float32) / 255.0
        alpha[protect[cy1:cy2, cx1:cx2] > 0] = 0.0
        alpha = alpha[..., None]

        out = image_bgr.copy()
        base = out[cy1:cy2, cx1:cx2].astype(np.float32)
        out[cy1:cy2, cx1:cx2] = (
            alpha * filled_crop.astype(np.float32) + (1.0 - alpha) * base
        ).astype(np.uint8)

        return out, halo, "lama"


# ----------------------------- standalone-отладка ----------------------------

def main() -> None:
    import torch

    cli = argparse.ArgumentParser(
        description="Заливка 'нимба' вокруг головы фоном (LaMa), этап 3.")
    cli.add_argument("--image", required=True,
                     help="Готовый кадр (candidate_*.png / best.png).")
    cli.add_argument("--region", required=True,
                     help="paste_mask.png из папки результата.")
    cli.add_argument("--out", default=None,
                     help="Куда сохранить (default: <image>_bgfill.png).")
    args = cli.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    region = cv2.imread(args.region, cv2.IMREAD_GRAYSCALE)
    if image is None or region is None:
        raise FileNotFoundError("Не удалось прочитать кадр или paste-маску.")
    if region.shape[:2] != image.shape[:2]:
        raise ValueError("Размеры кадра и paste-маски не совпадают.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    filler = BackgroundFiller(device)
    filled, halo, status = filler(image, region)

    out_path = Path(args.out) if args.out else \
        image_path.with_name(f"{image_path.stem}_bgfill.png")
    cv2.imwrite(str(out_path), filled)
    cv2.imwrite(str(out_path.with_name(out_path.stem + "_halo.png")), halo)
    print(f"Статус: {status}")
    print(f"Итог:   {out_path}")


if __name__ == "__main__":
    main()