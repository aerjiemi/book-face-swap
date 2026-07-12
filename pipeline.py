"""
pipeline.py — единая точка входа: иллюстрация + фото клиента -> N кандидатов.

Шаги (0-1 кэшируются, считаются один раз на сцену):
  0. Детекция лица персонажа на иллюстрации (RetinaFace) -> data/detections/
  1. Маска "лицо+волосы" персонажа (SAM2)                -> data/masks/
  2. Коллаж: волосы клиента на заготовку (hair_collage)
  3. SDXL inpaint + ControlNet Tile + IP-Adapter FaceID PlusV2, N сидов
  4. Заливка "нимба" вокруг новой головы (background_fill)
  5. Ранжирование по ArcFace-сходству: best.png, grid.jpg, result.json

Запуск:
  python pipeline.py --image data/illustrations/spread_01.png \
                     --client data/clients/ivan.jpg --seeds 5

Полезные флаги:
  --seeds N          сколько вариантов сгенерировать (default 5)
  --likeness         приоритет сходства с клиентом над стилем
  --face-index K     на сцене несколько лиц — заменить только K-е
  --force-detect / --force-mask   пересчитать кэш шагов 0-1
  --no-bg-fill       выключить заливку нимба
  остальные флаги тюнинга совпадают с output_inpainting.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import cv2
import numpy as np

# output_inpainting при импорте глушит логи библиотек (warnings/logging)
import output_inpainting as inp
from output_inpainting import quiet

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTIONS_DIR = PROJECT_ROOT / "data" / "detections"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"


# ============================ ШАГ 0: ДЕТЕКЦИЯ =================================

def ensure_detection(image_path: Path, threshold: float = 0.5,
                     force: bool = False) -> dict:
    """Детекция лица персонажа. Если JSON уже посчитан — кэш (заготовки
    статичны, шаг офлайн-подготовки книги). Тяжёлые импорты ленивые."""
    json_path = DETECTIONS_DIR / f"{image_path.stem}.json"
    if json_path.exists() and not force:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("detections"):
            print(f"[0] Детекция: кэш ({len(data['detections'])} лиц)")
            return data

    print("[0] Детекция лица персонажа (RetinaFace) ...")
    with quiet():
        from illustrations_detect import detect_faces, draw_visualization
        detections = detect_faces(image_path, threshold=threshold)
    if not detections:
        raise RuntimeError(
            f"На иллюстрации не найдено лицо: {image_path}. "
            f"Попробуйте --threshold 0.3 или другую сцену.")

    data = {"image": str(image_path), "threshold": threshold,
            "detections": detections}
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    draw_visualization(image_path, detections,
                       DETECTIONS_DIR / f"{image_path.stem}_vis.jpg")
    print(f"    найдено лиц: {len(detections)}")
    return data


# ============================ ШАГ 1: МАСКА (SAM2) =============================

def ensure_mask(image_path: Path, force: bool = False,
                face_index: int | None = None) -> Path:
    """Маска лицо+волосы персонажа. Кэшируется.

    face_index: замаскировать только одно лицо (индекс по убыванию score
    детектора). None = все лица на сцене.
    """
    stem = image_path.stem
    mask_path = MASKS_DIR / f"{stem}_mask.png"
    if mask_path.exists() and not force:
        print("[1] Маска: кэш")
        return mask_path

    print("[1] Маска лицо+волосы (SAM2) ...")
    import torch
    with quiet():
        import illustrations_mask as im

    detection = im.load_detection(stem)
    dets = detection["detections"]
    if face_index is not None:
        if not (0 <= face_index < len(dets)):
            raise ValueError(f"--face-index {face_index}: на сцене "
                             f"{len(dets)} лиц (индексы 0..{len(dets) - 1}).")
        dets = [dets[face_index]]

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Не удалось прочитать: {image_path}")
    img_h, img_w = image_bgr.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with quiet():
        predictor = im.build_predictor(im.DEFAULT_MODEL, device)
        predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    combined = np.zeros((img_h, img_w), dtype=bool)
    boxes, points_by_face = [], []
    for det in dets:
        box = im.expand_box(det["bbox"], img_w, img_h, im.DEFAULT_PAD_UP,
                            im.DEFAULT_PAD_DOWN, im.DEFAULT_PAD_SIDE)
        facial_pts = [pt for pt in det["landmarks"].values()]
        hair_pts = im.hair_points(det["bbox"], box, im.DEFAULT_HAIR_POINTS)
        points = facial_pts + hair_pts
        mask, _score = im.predict_mask(predictor, box, points,
                                       [1] * len(points), device)
        combined |= im.clean_mask(mask, im.DEFAULT_CLOSE_KERNEL,
                                  im.DEFAULT_MIN_COMP_FRAC)
        boxes.append(box)
        points_by_face.append({"facial": facial_pts, "hair": hair_pts})

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), combined.astype(np.uint8) * 255)
    im.draw_visualization(image_bgr, combined, boxes, points_by_face,
                          MASKS_DIR / f"{stem}_vis.jpg")

    # освобождаем VRAM под SDXL
    del predictor
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"    маска сохранена: {mask_path.name} "
          f"(проверьте {stem}_vis.jpg глазами)")
    return mask_path


# ==================== ШАГИ 2-5: КОЛЛАЖ + ИНПЕЙНТ + НИМБ =======================

def generate(image_path: Path, client_path: Path, cfg,
             paste_face: bool = False) -> dict:
    """Генерация cfg.seeds кандидатов. Возвращает dict с путями и метриками.

    Логика идентична output_inpainting.py; face-parsing грузится один раз
    и переиспользуется коллажом и заливкой нимба.
    """
    import torch
    from PIL import Image
    from hair_collage import (build_face_parser, client_hair_and_face_masks,
                              character_hair_face_on_illustration,
                              align_transform, build_collage)
    from background_fill import BackgroundFiller

    stem = image_path.stem
    mask_path = MASKS_DIR / f"{stem}_mask.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # --- identity клиента ---
    face_app = inp.build_face_analyzer()
    client_bgr, cface = inp.client_face(face_app, client_path)
    ref_embed = cface.normed_embedding.astype(np.float32)

    # --- иллюстрация + маска ---
    illus_bgr = cv2.imread(str(image_path))
    mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if illus_bgr is None or mask_full is None:
        raise ValueError("Не удалось прочитать иллюстрацию или маску.")
    if mask_full.shape[:2] != illus_bgr.shape[:2]:
        raise ValueError("Размеры маски и иллюстрации не совпадают.")
    img_h, img_w = illus_bgr.shape[:2]

    # --- шаг 2: коллаж ---
    print("[2] Коллаж волос клиента ...")
    with quiet():
        parser = build_face_parser(device)
        hair_m, face_m = client_hair_and_face_masks(client_bgr, parser)
        char_hair, _cf, ok = character_hair_face_on_illustration(
            illus_bgr, mask_full, parser)
        M, align_method, char_bbox = align_transform(
            face_app, client_bgr, cface, illus_bgr, mask_full)
        col = build_collage(illus_bgr, mask_full, client_bgr, hair_m, face_m,
                            M, align_method, char_hair=char_hair,
                            char_parse_ok=ok, char_face_bbox=char_bbox,
                            paste_face=paste_face)
    print(f"    выравнивание={col.align_method}, "
          f"волосы={'да' if col.hair_pasted else 'нет'}, "
          f"затирание={col.wipe_method}")

    inpaint_region = col.inpaint_region.copy()
    x1, y1, x2, y2 = inp.mask_crop_box(inpaint_region, inp.CROP_MARGIN,
                                       img_w, img_h)
    side = x2 - x1

    gen_seam = max(1, round(cfg.seam_dilate * side / cfg.gen_size))
    paste_seam = max(0, round(cfg.paste_dilate * side / cfg.gen_size))
    gen_mask_full = inp.dilate_px(inpaint_region, gen_seam)
    paste_mask_full = inp.dilate_px(inpaint_region, paste_seam)

    crop_paste_mask = paste_mask_full[y1:y2, x1:x2]
    gs = cfg.gen_size
    crop_rgb = cv2.resize(
        cv2.cvtColor(col.collage_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB),
        (gs, gs), interpolation=cv2.INTER_LANCZOS4)
    mask_gs = cv2.resize(gen_mask_full[y1:y2, x1:x2], (gs, gs),
                         interpolation=cv2.INTER_NEAREST)
    if cfg.mask_blur > 0:
        k = cfg.mask_blur * 2 + 1
        mask_gs = cv2.GaussianBlur(mask_gs, (k, k), 0)

    pil_image = Image.fromarray(crop_rgb)
    pil_control = Image.fromarray(crop_rgb)
    pil_mask = Image.fromarray(mask_gs)

    # --- шаг 3: модели ---
    print("[3] Загрузка моделей (SDXL + ControlNet + FaceID) ...")
    pipe = inp.build_pipeline(device, cfg)
    pipe.set_ip_adapter_scale(cfg.ip_scale)
    if cfg.faceid_version == "plusv2":
        inp.setup_faceid_plusv2(pipe, client_bgr, cface, device, dtype)
    id_embeds = inp.faceid_embeds_for_pipe(ref_embed, device, dtype)

    bg_filler = None
    if cfg.bg_fill:
        bg_filler = BackgroundFiller(device, pipe=pipe, gen_size=cfg.gen_size,
                                     control_scale=cfg.control_scale,
                                     restore_ip_scale=cfg.ip_scale,
                                     parser=parser)   # переиспользуем парсер
        bg_filler.set_ref_embed_shape(tuple(id_embeds.shape))

    out_dir = OUTPUTS_DIR / f"{stem}__{client_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "collage.png"), col.collage_bgr)
    cv2.imwrite(str(out_dir / "inpaint_mask.png"), gen_mask_full)
    cv2.imwrite(str(out_dir / "paste_mask.png"), paste_mask_full)

    blur_px = max(1, round(cfg.mask_blur * side / cfg.gen_size))
    kb = 2 * blur_px + 1
    alpha_full = cv2.GaussianBlur(crop_paste_mask, (kb, kb), 0)
    alpha_full = (alpha_full.astype(np.float32) / 255.0)[..., None]

    # --- шаг 4-5: генерация кандидатов ---
    print(f"[4] Генерация {cfg.seeds} кандидатов ...")
    candidates = []
    for i in range(cfg.seeds):
        seed = cfg.base_seed + i
        gen = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.time()
        result = pipe(
            prompt=cfg.prompt, negative_prompt=cfg.negative,
            image=pil_image, mask_image=pil_mask, control_image=pil_control,
            controlnet_conditioning_scale=cfg.control_scale,
            ip_adapter_image_embeds=[id_embeds],
            strength=cfg.strength, num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance, generator=gen,
            height=gs, width=gs,
        ).images[0]

        gen_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.resize(gen_bgr, (side, side),
                             interpolation=cv2.INTER_LANCZOS4)
        composed = illus_bgr.copy()
        region = composed[y1:y2, x1:x2].astype(np.float32)
        blended = (alpha_full * gen_bgr.astype(np.float32)
                   + (1 - alpha_full) * region)
        composed[y1:y2, x1:x2] = blended.astype(np.uint8)

        bg_status, halo_mask = "disabled", None
        if bg_filler is not None:
            with quiet():
                composed, halo_mask, bg_status = bg_filler(
                    composed, paste_mask_full, inpaint_region)

        sim = inp.face_similarity(face_app, composed[y1:y2, x1:x2], ref_embed)
        dt = time.time() - t0
        print(f"    {i + 1}/{cfg.seeds}  seed={seed}  similarity={sim:.2f}"
              f"  нимб={bg_status}  ({dt:.1f} c)"
              + ("  [лицо не найдено!]" if sim < 0 else ""))

        cand_path = out_dir / f"candidate_{seed}.png"
        cv2.imwrite(str(cand_path), composed)
        candidates.append({"seed": seed, "similarity": sim,
                           "image_bgr": composed, "path": str(cand_path),
                           "bg_status": bg_status, "halo_mask": halo_mask,
                           "seconds": round(dt, 1)})

    best = max(candidates, key=lambda c: c["similarity"])
    best_path = out_dir / "best.png"
    cv2.imwrite(str(best_path), best["image_bgr"])
    if best.get("halo_mask") is not None:
        cv2.imwrite(str(out_dir / "bg_fill_mask.png"), best["halo_mask"])

    if best["similarity"] < 0.15:
        print("\n[!] ВНИМАНИЕ: на лучшем кандидате лицо почти не "
              "детектируется. Попробуйте --control-scale 0.5 и/или "
              "--ip-scale 1.0; wipe_method='geometric' — признак неточной "
              "маски головы (пересчитайте с --force-mask).")

    grid_path = out_dir / "grid.jpg"
    inp.make_grid(candidates, (x1, y1, x2, y2), grid_path)

    result_meta = {
        "variant": "pipeline_v1 (collage+controlnet+plusv2+bgfill)",
        "image": str(image_path), "client": str(client_path),
        "mask": str(mask_path), "crop_box": [x1, y1, x2, y2],
        "faceid_version": cfg.faceid_version,
        "base_model": cfg.base_model or inp.SDXL_INPAINT_ID,
        "align_method": col.align_method, "wipe_method": col.wipe_method,
        "hair_pasted": col.hair_pasted, "bg_fill": cfg.bg_fill,
        "prompt": cfg.prompt, "negative": cfg.negative,
        "ip_scale": cfg.ip_scale,
        "faceid_lora_scale": cfg.faceid_lora_scale,
        "control_scale": cfg.control_scale,
        "strength": cfg.strength, "steps": cfg.steps,
        "guidance": cfg.guidance, "gen_size": cfg.gen_size,
        "seam_dilate": cfg.seam_dilate, "paste_dilate": cfg.paste_dilate,
        "candidates": [{"seed": c["seed"], "similarity": c["similarity"],
                        "path": c["path"], "bg_fill": c["bg_status"],
                        "seconds": c["seconds"]} for c in candidates],
        "best_seed": best["seed"], "best_similarity": best["similarity"],
        "best_path": str(best_path),
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_meta, f, ensure_ascii=False, indent=2)

    print(f"\nЛучший seed: {best['seed']} (similarity={best['similarity']:.2f})")
    print(f"Итог:  {best_path}")
    print(f"Сетка: {grid_path}")
    return result_meta


# ================================ ОРКЕСТРАЦИЯ =================================

def run_pipeline(image: str | Path, client: str | Path, *,
                 seeds: int | None = None, likeness: bool = False,
                 face_index: int | None = None,
                 force_detect: bool = False, force_mask: bool = False,
                 threshold: float = 0.5, paste_face: bool = False,
                 cfg_overrides: dict | None = None) -> dict:
    """Полный прогон: детекция -> маска -> коллаж -> инпейнт -> нимб -> выбор.
    Возвращает result.json-словарь (пути кандидатов в поле "candidates")."""
    image_path, client_path = Path(image), Path(client)
    for p, what in [(image_path, "иллюстрация"), (client_path, "фото клиента")]:
        if not p.exists():
            raise FileNotFoundError(f"Не найден файл ({what}): {p}")

    overrides = dict(cfg_overrides or {})
    if seeds is not None:
        overrides["seeds"] = seeds
    if likeness:
        preset = {"prompt": inp.LIKENESS_PROMPT,
                  "negative": inp.LIKENESS_NEGATIVE,
                  "control_scale": 0.18, "strength": 1.0,
                  "ip_scale": 1.0, "guidance": 5.0}
        for k, v in preset.items():
            overrides.setdefault(k, v)
    cfg = dataclasses.replace(inp.CFG, **overrides)

    print(f"Сцена: {image_path.name} | Клиент: {client_path.name}"
          + ("  [likeness]" if likeness else ""))
    t_start = time.time()
    ensure_detection(image_path, threshold=threshold, force=force_detect)
    ensure_mask(image_path, force=force_mask, face_index=face_index)
    result = generate(image_path, client_path, cfg, paste_face=paste_face)
    result["total_seconds"] = round(time.time() - t_start, 1)
    print(f"Полное время: {result['total_seconds']} c "
          f"(включая загрузку моделей)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end пайплайн: иллюстрация + фото клиента -> "
                    "N кандидатов с лицом клиента.")
    parser.add_argument("--image", required=True, help="Иллюстрация-заготовка.")
    parser.add_argument("--client", required=True, help="Фото клиента.")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Сколько вариантов сгенерировать (default 5).")
    parser.add_argument("--likeness", action="store_true",
                        help="Приоритет сходства с клиентом над стилем.")
    parser.add_argument("--face-index", type=int, default=None,
                        help="На сцене несколько лиц — заменить только это "
                             "(индекс из детекции). По умолчанию все.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Порог RetinaFace на иллюстрации.")
    parser.add_argument("--force-detect", action="store_true",
                        help="Пересчитать детекцию, игнорируя кэш.")
    parser.add_argument("--force-mask", action="store_true",
                        help="Пересчитать маску SAM2, игнорируя кэш.")
    parser.add_argument("--no-bg-fill", action="store_true",
                        help="Отключить заливку нимба.")
    parser.add_argument("--paste-face", action="store_true",
                        help="Вклеивать в коллаж и лицо клиента (эксперимент).")
    parser.add_argument("--faceid", default=None, choices=["v1", "plusv2"])
    parser.add_argument("--base-model", default=None,
                        help="Другой SDXL-чекпоинт (HF id или путь).")
    parser.add_argument("--ip-scale", type=float, default=None)
    parser.add_argument("--faceid-lora-scale", type=float, default=None)
    parser.add_argument("--control-scale", type=float, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--paste-dilate", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None,
                        help="Стартовый seed (default 42).")
    args = parser.parse_args()

    cfg_overrides = {}
    mapping = {"ip_scale": args.ip_scale,
               "faceid_lora_scale": args.faceid_lora_scale,
               "control_scale": args.control_scale,
               "strength": args.strength, "steps": args.steps,
               "guidance": args.guidance, "paste_dilate": args.paste_dilate,
               "base_seed": args.base_seed, "faceid_version": args.faceid,
               "base_model": args.base_model}
    for k, v in mapping.items():
        if v is not None:
            cfg_overrides[k] = v
    if args.no_bg_fill:
        cfg_overrides["bg_fill"] = False

    run_pipeline(args.image, args.client, seeds=args.seeds,
                 likeness=args.likeness, face_index=args.face_index,
                 force_detect=args.force_detect, force_mask=args.force_mask,
                 threshold=args.threshold, paste_face=args.paste_face,
                 cfg_overrides=cfg_overrides)


if __name__ == "__main__":
    main()