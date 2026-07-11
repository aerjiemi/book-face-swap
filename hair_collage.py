"""
Этап 1 варианта 3 (двухэтапный пайплайн): КОЛЛАЖ волос клиента на заготовку.

Идея: геометрию и цвет причёски мы НЕ доверяем диффузии, а закладываем руками
до инпейнта. Тогда на этапе 2 модель должна только "перерисовать в стиле",
а не "придумать волосы".

Что делает модуль:
1) Сегментация волос на фото клиента.
   Используется face-parsing (SegFormer, класс "hair"), а не SAM:
   SAM не знает семантику "волосы" и требует точек-подсказок,
   face-parsing выдаёт маску волос напрямую и очень надёжен на
   качественных фото клиентов. Работает и для лысых/коротких стрижек
   (маска просто будет маленькой — это обрабатывается).
2) Выравнивание головы клиента к голове персонажа:
   5 точек лица клиента (InsightFace kps) -> 5 точек лица персонажа,
   similarity-преобразование (масштаб + поворот + сдвиг),
   cv2.estimateAffinePartial2D. Если лицо персонажа на иллюстрации
   не детектируется — грубый fallback по bbox предрасчитанной маски.
3) "Чистая заготовка": область исходных лица+волос персонажа затирается
   cv2.inpaint (Telea). Это важно: иначе ControlNet на этапе 2 будет
   "видеть" СТАРЫЕ волосы персонажа и тянуть их обратно в кадр.
4) Волосы клиента (фото-пиксели) вклеиваются поверх чистой заготовки
   с мягким краем. Опционально вклеивается и лицо (--paste-face) —
   может помочь ControlNet с геометрией, но обычно лицо лучше оставить
   целиком на откуп IP-Adapter FaceID.

Standalone-отладка (сохраняет collage/маски в data/outputs/_collage_debug):
  python hair_collage.py --image data/illustrations/spread_01.png \
                         --client data/clients/ivan.jpg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FACE_PARSING_ID = "jonathandinu/face-parsing"

# имена классов из id2label модели face-parsing (CelebAMask-HQ)
HAIR_LABELS = {"hair"}
FACE_LABELS = {"skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow",
               "eye_g", "u_lip", "l_lip", "mouth", "l_ear", "r_ear"}

# если волос на фото меньше этой доли площади bbox лица —
# считаем клиента лысым/очень коротко стриженным и коллаж волос не делаем
MIN_HAIR_AREA_FRAC = 0.05


@dataclass
class CollageResult:
    collage_bgr: np.ndarray        # заготовка: живописное лицо персонажа + волосы клиента
    inpaint_region: np.ndarray     # что перерисовывать (лицо+волосы+волосы клиента), 0/255
    warped_hair_mask: np.ndarray   # маска вклеенных волос клиента (для отладки), 0/255
    face_preserved: bool           # оставили ли живописное лицо персонажа как style-ref
    wipe_method: str               # "parse" | "face_bbox" | "geometric"
    hair_pasted: bool              # вклеивали ли волосы (False для лысых / фолбэков)
    align_method: str              # "kps" | "bbox_fallback"


# ----------------------------- сегментация (face-parsing) --------------------

def build_face_parser(device: str):
    """SegFormer face-parsing: возвращает (processor, model, id2label)."""
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    processor = SegformerImageProcessor.from_pretrained(FACE_PARSING_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(FACE_PARSING_ID)
    model.to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return processor, model, id2label


def _parse_labels(image_bgr: np.ndarray, parser) -> np.ndarray:
    """Карта классов face-parsing в разрешении исходного изображения (uint8)."""
    import torch

    processor, model, _ = parser
    device = next(model.parameters()).device
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits            # (1, C, h/4, w/4)
    up = torch.nn.functional.interpolate(
        logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
    )
    return up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def _labels_to_mask(labels_map: np.ndarray, id2label: dict, wanted: set[str]) -> np.ndarray:
    wanted_ids = {i for i, name in id2label.items() if name in wanted}
    mask = np.isin(labels_map, list(wanted_ids)).astype(np.uint8) * 255
    return mask


def _keep_main_components(mask: np.ndarray, min_frac: float = 0.02) -> np.ndarray:
    """Убирает мелкий мусор: оставляет компоненты крупнее min_frac от максимальной."""
    num, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.where(areas >= areas.max() * min_frac)[0] + 1
    out = np.isin(lab, keep).astype(np.uint8) * 255
    return out


def client_hair_and_face_masks(image_bgr: np.ndarray, parser
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Маски волос и лица клиента (0/255) в координатах его фото."""
    labels_map = _parse_labels(image_bgr, parser)
    id2label = parser[2]
    hair = _keep_main_components(_labels_to_mask(labels_map, id2label, HAIR_LABELS))
    face = _keep_main_components(_labels_to_mask(labels_map, id2label, FACE_LABELS))
    # лёгкое закрытие дыр (блики на волосах и т.п.)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, kernel)
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, kernel)
    return hair, face


def character_hair_face_on_illustration(illus_bgr: np.ndarray, mask_full: np.ndarray,
                                        parser) -> tuple[np.ndarray, np.ndarray, bool]:
    """Маски ВОЛОС и ЛИЦА персонажа на самой иллюстрации (вариант B).

    Нужно, чтобы затирать только волосы персонажа, а живописное лицо
    оставлять как texture-reference для ControlNet Tile.

    Парсинг делаем на кропе вокруг предрасчитанной маски головы (zoom
    повышает надёжность SegFormer на стилизованном рисунке), потом мапим
    обратно и ограничиваем результат областью mask_full.

    Возвращает (char_hair, char_face, ok). ok=False, если парсинг на
    иллюстрации не дал осмысленной маски волос — тогда вызывающий код
    откатывается к затиранию всей головы (старое поведение).
    """
    h, w = illus_bgr.shape[:2]
    ys, xs = np.where(mask_full > 0)
    if len(xs) == 0:
        return np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), False

    # кроп вокруг головы с запасом
    pad = int(0.25 * max(xs.max() - xs.min(), ys.max() - ys.min()))
    cx1 = max(0, xs.min() - pad); cy1 = max(0, ys.min() - pad)
    cx2 = min(w, xs.max() + pad); cy2 = min(h, ys.max() + pad)
    crop = illus_bgr[cy1:cy2, cx1:cx2]

    hair_c, face_c = client_hair_and_face_masks(crop, parser)

    char_hair = np.zeros((h, w), np.uint8)
    char_face = np.zeros((h, w), np.uint8)
    char_hair[cy1:cy2, cx1:cx2] = hair_c
    char_face[cy1:cy2, cx1:cx2] = face_c

    # держим маски строго внутри известной области головы (mask_full + запас)
    guard = cv2.dilate(mask_full,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    char_hair = cv2.bitwise_and(char_hair, guard)
    char_face = cv2.bitwise_and(char_face, guard)

    # парсинг считаем успешным, если волосы заняли заметную часть головы
    head_area = max(int((mask_full > 0).sum()), 1)
    ok = int((char_hair > 0).sum()) > 0.03 * head_area
    return char_hair, char_face, ok


# ----------------------------- выравнивание ----------------------------------

def _best_face_for_mask(faces, mask_full: np.ndarray):
    """Из лиц, найденных на иллюстрации, берём то, чей bbox лучше пересекается
    с предрасчитанной маской лица+волос персонажа."""
    ys, xs = np.where(mask_full > 0)
    mx1, mx2, my1, my2 = xs.min(), xs.max(), ys.min(), ys.max()

    def overlap(f):
        x1, y1, x2, y2 = f.bbox
        ix = max(0, min(x2, mx2) - max(x1, mx1))
        iy = max(0, min(y2, my2) - max(y1, my1))
        return ix * iy

    faces = [f for f in faces if overlap(f) > 0]
    if not faces:
        return None
    return max(faces, key=overlap)


def align_transform(face_app, client_bgr: np.ndarray, client_face,
                    illus_bgr: np.ndarray, mask_full: np.ndarray
                    ) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Similarity-преобразование "фото клиента -> иллюстрация".

    Основной путь: 5 kps клиента -> 5 kps персонажа.
    Fallback (лицо на иллюстрации не нашлось): масштабируем bbox лица клиента
    в нижние ~2/3 bbox маски (лицо обычно под волосами) — грубо, но рабоче.

    Возвращает (M, method, char_face_bbox). char_face_bbox — bbox лица
    персонажа на иллюстрации (или None), нужен чтобы защитить лицо от
    затирания в коллаже даже когда парсинг волос провалился.
    """
    illus_faces = face_app.get(illus_bgr)
    target = _best_face_for_mask(illus_faces, mask_full) if illus_faces else None
    char_bbox = target.bbox.astype(np.float32) if target is not None else None

    if target is not None:
        src = client_face.kps.astype(np.float32)
        dst = target.kps.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is not None:
            return M.astype(np.float32), "kps", char_bbox

    # --- fallback по bbox маски ---
    ys, xs = np.where(mask_full > 0)
    mx1, mx2, my1, my2 = xs.min(), xs.max(), ys.min(), ys.max()
    mw, mh = mx2 - mx1, my2 - my1
    cb = client_face.bbox  # x1,y1,x2,y2
    cw, ch = cb[2] - cb[0], cb[3] - cb[1]
    # лицо персонажа ~ 55% высоты маски, центр — в нижней части маски
    scale = (0.55 * mh) / max(ch, 1)
    tx = (mx1 + 0.5 * mw) - scale * (cb[0] + 0.5 * cw)
    ty = (my1 + 0.62 * mh) - scale * (cb[1] + 0.5 * ch)
    M = np.array([[scale, 0, tx], [0, scale, 0.0 + ty]], np.float32)
    print("[hair_collage] ВНИМАНИЕ: лицо персонажа на иллюстрации не "
          "детектировано, использую грубое выравнивание по bbox маски.")
    return M, "bbox_fallback", char_bbox


# ----------------------------- коллаж ----------------------------------------

def hair_wipe_region(mask_full: np.ndarray, char_hair: np.ndarray | None,
                     char_parse_ok: bool, char_face_bbox: np.ndarray | None
                     ) -> tuple[np.ndarray, str]:
    """Регион для затирания = ТОЛЬКО волосы персонажа. Лицо всегда защищено.

    Три уровня надёжности (лицо не затирается ни в одном):
      1) "parse"     — face-parsing выделил волосы персонажа (лучший случай);
      2) "face_bbox" — парсинг не сработал, но InsightFace нашёл лицо персонажа:
                       защищаем прямоугольник лица, волосы = маска минус лицо;
      3) "geometric" — не сработало ничего: защищаем нижние ~60% маски головы
                       (там обычно лицо), затираем только верхнюю часть (волосы).

    Именно уровни 2-3 закрывают старый баг "лицо слилось с фоном": раньше при
    провале парсинга затиралась ВСЯ голова, и модель дорисовывала лицо из
    контекста (динозавр/лес). Теперь живописное лицо остаётся как ref для Tile.
    """
    h, w = mask_full.shape[:2]
    kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    if char_parse_ok and char_hair is not None and char_hair.any():
        return cv2.dilate(char_hair, kernel15), "parse"

    # строим маску защищаемого лица
    protect = np.zeros((h, w), np.uint8)
    if char_face_bbox is not None:
        fx1, fy1, fx2, fy2 = [int(v) for v in char_face_bbox]
        px = int(0.15 * max(fx2 - fx1, 1)); py = int(0.15 * max(fy2 - fy1, 1))
        cv2.rectangle(protect, (fx1 - px, fy1 - py), (fx2 + px, fy2 + py), 255, -1)
        method = "face_bbox"
    else:
        ys, xs = np.where(mask_full > 0)
        my1, my2 = ys.min(), ys.max()
        cut = int(my1 + 0.40 * (my2 - my1))   # выше этой линии = волосы
        protect[cut:, :] = 255
        method = "geometric"
    protect = cv2.bitwise_and(protect, mask_full)

    # волосы = голова минус лицо; дилейтим, но лицо оставляем чистым
    wipe = cv2.subtract(mask_full, protect)
    wipe = cv2.dilate(wipe, kernel15)
    wipe = cv2.subtract(wipe, cv2.erode(protect, kernel9))  # не заезжать на лицо
    print(f"[hair_collage] волосы персонажа не сегментированы парсингом, "
          f"защищаю лицо методом '{method}' (лицо НЕ затирается).")
    return wipe, method


def build_collage(illus_bgr: np.ndarray, mask_full: np.ndarray,
                  client_bgr: np.ndarray, client_hair: np.ndarray,
                  client_face_mask: np.ndarray, M: np.ndarray,
                  align_method: str,
                  char_hair: np.ndarray | None = None,
                  char_parse_ok: bool = False,
                  char_face_bbox: np.ndarray | None = None,
                  paste_face: bool = False,
                  feather_px: int = 5) -> CollageResult:
    """Заготовка с ЖИВОПИСНЫМ лицом персонажа + волосы клиента поверх.

    Вариант B: НЕ затираем лицо персонажа (оно нужно ControlNet Tile как
    texture-reference — иначе лицо перерисовывается в гладкий "пластик" или
    сливается с фоном). Затираем ТОЛЬКО волосы персонажа.
    """
    h, w = illus_bgr.shape[:2]

    wipe, wipe_method = hair_wipe_region(mask_full, char_hair, char_parse_ok,
                                         char_face_bbox)
    face_preserved = True   # лицо персонажа сохраняется при любом wipe_method
    clean = cv2.inpaint(illus_bgr, wipe, 5, cv2.INPAINT_TELEA)

    # регион, который на этапе 2 будет перерисован (лицо+волосы+волосы клиента)
    inpaint_region = mask_full.copy()

    # клиент лысый / волос почти нет -> коллаж волос пропускаем
    faces_area = max(int((client_face_mask > 0).sum()), 1)
    hair_area = int((client_hair > 0).sum())
    if hair_area < MIN_HAIR_AREA_FRAC * faces_area:
        print("[hair_collage] Волос на фото клиента почти нет — коллаж "
              "пропущен, причёску опишите через --hair.")
        return CollageResult(clean, inpaint_region, np.zeros((h, w), np.uint8),
                             face_preserved, wipe_method, False, align_method)

    # варпим фото и маски клиента в координаты иллюстрации
    paste_mask = client_hair.copy()
    if paste_face:
        paste_mask = np.maximum(paste_mask, client_face_mask)

    warped_photo = cv2.warpAffine(client_bgr, M, (w, h), flags=cv2.INTER_LANCZOS4)
    warped_paste = cv2.warpAffine(paste_mask, M, (w, h), flags=cv2.INTER_LINEAR)
    warped_hair = cv2.warpAffine(client_hair, M, (w, h), flags=cv2.INTER_LINEAR)
    warped_paste = ((warped_paste > 127).astype(np.uint8)) * 255
    warped_hair = ((warped_hair > 127).astype(np.uint8)) * 255

    # мягкая вклейка фото-пикселей поверх заготовки (лицо персонажа остаётся)
    k = feather_px * 2 + 1
    alpha = cv2.GaussianBlur(warped_paste, (k, k), 0).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    collage = (alpha * warped_photo.astype(np.float32)
               + (1.0 - alpha) * clean.astype(np.float32)).astype(np.uint8)

    # волосы клиента могут выходить за исходную маску головы -> расширяем регион
    inpaint_region = np.maximum(inpaint_region, warped_hair)

    return CollageResult(collage, inpaint_region, warped_hair,
                         face_preserved, wipe_method, True, align_method)


def make_collage_for(illus_bgr: np.ndarray, mask_full: np.ndarray,
                     client_bgr: np.ndarray, face_app, client_face,
                     device: str, paste_face: bool = False) -> CollageResult:
    """Полный этап 1: сегментация клиента + персонажа -> выравнивание -> коллаж."""
    parser = build_face_parser(device)
    hair, face_mask = client_hair_and_face_masks(client_bgr, parser)
    char_hair, _char_face, ok = character_hair_face_on_illustration(
        illus_bgr, mask_full, parser)
    M, method, char_bbox = align_transform(face_app, client_bgr, client_face,
                                           illus_bgr, mask_full)
    return build_collage(illus_bgr, mask_full, client_bgr, hair, face_mask,
                         M, method, char_hair=char_hair, char_parse_ok=ok,
                         char_face_bbox=char_bbox, paste_face=paste_face)


# ----------------------------- standalone-отладка ----------------------------

def main() -> None:
    import torch
    from illustrations_mask import PROJECT_ROOT, MASKS_DIR

    parser_cli = argparse.ArgumentParser(description="Отладка коллажа волос.")
    parser_cli.add_argument("--image", required=True)
    parser_cli.add_argument("--client", required=True)
    parser_cli.add_argument("--paste-face", action="store_true")
    args = parser_cli.parse_args()

    image_path, client_path = Path(args.image), Path(args.client)
    mask_path = MASKS_DIR / f"{image_path.stem}_mask.png"
    illus = cv2.imread(str(image_path))
    client = cv2.imread(str(client_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if illus is None or client is None or mask is None:
        raise FileNotFoundError("Не удалось прочитать иллюстрацию/фото/маску.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(client)
    if not faces:
        raise ValueError("На фото клиента не найдено лицо.")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)

    res = make_collage_for(illus, mask, client, app, faces[0], device,
                           paste_face=args.paste_face)

    out_dir = PROJECT_ROOT / "data" / "outputs" / "_collage_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{image_path.stem}__{client_path.stem}_collage.png"),
                res.collage_bgr)
    cv2.imwrite(str(out_dir / f"{image_path.stem}__{client_path.stem}_hair.png"),
                res.warped_hair_mask)
    cv2.imwrite(str(out_dir / f"{image_path.stem}__{client_path.stem}_region.png"),
                res.inpaint_region)
    print(f"Коллаж: {out_dir} (выравнивание: {res.align_method}, "
          f"волосы вклеены: {res.hair_pasted}, "
          f"лицо-персонажа сохранено: {res.face_preserved}, "
          f"метод затирания волос: {res.wipe_method})")


if __name__ == "__main__":
    main()