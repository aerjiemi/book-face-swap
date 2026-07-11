"""
Этап 3 пайплайна, ВАРИАНТ 3: двухэтапный пайплайн (коллаж -> инпейнт + ControlNet).

За основу взята версия "до" (обычный IP-Adapter FaceID, без PlusV2):
она стабильно рисовала лицо, проблема была только с волосами.

Как работает:
1) КОЛЛАЖ (hair_collage.py):
   - волосы клиента сегментируются face-parsing'ом и вырезаются с его фото;
   - голова клиента выравнивается к голове персонажа по 5 точкам лица
     (similarity: масштаб+поворот+сдвиг);
   - исходные лицо+волосы персонажа затираются (cv2.inpaint), чтобы старая
     причёска персонажа не "просачивалась" через ControlNet;
   - волосы клиента (фото-пиксели) вклеиваются поверх.
   Теперь ГЕОМЕТРИЯ и ЦВЕТ причёски заданы жёстко — их не надо "угадывать".

2) ИНПЕЙНТ: SDXL-inpaint + IP-Adapter FaceID (v1, только ArcFace)
   + ControlNet Tile (xinsir) с малым весом (по умолчанию 0.35) на всю зону.
   - базовое изображение и control image = коллаж;
   - маска инпейнта = затёртая зона + вклеенные волосы (+ запас на швы);
   - strength=0.99: зона перерисовывается полностью, но ControlNet Tile
     удерживает форму/цвет вклеенных волос, а стиль (мазки, палитра)
     задаётся промптом и style LoRA. Лицо ControlNet почти не ограничивает
     (в коллаже там затёртое пятно) — его рисует FaceID.

3) Отбор: несколько seed, лучший по ArcFace-сходству с клиентом.
   best.png / candidate_<seed>.png / grid.jpg / collage.png /
   inpaint_mask.png / result.json

Запуск (RTX 5050 8GB: работает через cpu-offload, медленно; на 4090 быстро):
  python output_inpainting_v3.py --image data/illustrations/spread_01.png \
      --client data/clients/ivan.jpg --hair "long wavy blonde hair"

  --hair          текст про причёску (англ.) — опционально, но помогает;
  --control-scale вес ControlNet Tile (0.3..0.45); больше = точнее форма
                  волос, но фактура остаётся "фотографичной";
  --paste-face    вклеивать в коллаж и лицо клиента (эксперимент);
  --gen-size 768  если на 8GB ловите OOM (1024 по умолчанию — SDXL обучен
                  на 1024, на 512 лица получаются кривыми, это и сломало
                  прошлую версию).

!!! Требует предрасчитанной маски (illustrations_mask.py).
Первый запуск докачает ControlNet Tile (~2.5GB) и face-parsing (~340MB).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from illustrations_mask import PROJECT_ROOT, ILLUSTRATIONS_DIR, MASKS_DIR
from hair_collage import make_collage_for

OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_TILE_ID = "xinsir/controlnet-tile-sdxl-1.0"
FACEID_REPO = "h94/IP-Adapter-FaceID"
FACEID_WEIGHTS = "ip-adapter-faceid_sdxl.bin"          # обычный FaceID (v1)
FACEID_LORA = "ip-adapter-faceid_sdxl_lora.safetensors"

CROP_MARGIN = 0.4   # запас контекста вокруг зоны инпейнта (доля от её размера)

DEFAULT_PROMPT = (
    "oil painting portrait of a person, painted in the same style as the artwork, "
    "visible brush strokes, thick impasto, textured painterly skin, "
    "hair shape and hair color exactly as in the image, "
    "accurate likeness of the reference face, "
    "palette consistent with the artwork, seamlessly integrated into the scene"
)
DEFAULT_NEGATIVE = (
    "photo, photorealistic, photographic skin, photographic hair texture, "
    "smooth skin, airbrushed, plastic skin, 3d render, vector, flat shading, "
    "deformed, blurry, low quality, extra eyes, bad anatomy, "
    "text, watermark, different person, generic face"
)


# ----------------------------- настройки инпейнта ----------------------------

@dataclass(frozen=True)
class InpaintConfig:
    lora_path: str | None = None   # путь к style LoRA
    lora_scale: float = 0.5
    ip_scale: float = 0.8          # сила identity (FaceID), 0.6..1.0; ниже = меньше "пластика"
    faceid_lora_scale: float = 0.6 # вес вспомогательной LoRA FaceID (была захардкожена 1.0)
    control_scale: float = 0.45    # вес ControlNet Tile, 0.35..0.55; выше = больше мазков/структуры
    strength: float = 0.99         # 1.0 = полностью перерисовать зону
    steps: int = 30
    guidance: float = 5.0
    seeds: int = 3                 # сколько кандидатов сгенерировать
    base_seed: int = 42
    gen_size: int = 1024           # рабочее разрешение (SDXL обучен на 1024!)
    mask_blur: int = 8             # размытие краёв маски (px в кропе gen_size)
    seam_dilate: int = 16          # запас маски на швы (px в кропе gen_size)
    prompt: str = DEFAULT_PROMPT
    negative: str = DEFAULT_NEGATIVE


CFG = InpaintConfig()


# ----------------------------- identity-embedding ----------------------------

def build_face_analyzer():
    """InsightFace buffalo_l: детекция + ArcFace-эмбеддинг."""
    from insightface.app import FaceAnalysis
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available() else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def client_face(app, photo_path: Path):
    """Крупнейшее лицо на фото клиента (kps нужны для выравнивания коллажа)."""
    img = cv2.imread(str(photo_path))
    if img is None:
        raise ValueError(f"Не удалось прочитать фото клиента: {photo_path}")
    faces = app.get(img)
    if not faces:
        raise ValueError(f"На фото клиента не найдено лицо: {photo_path}.")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    return img, faces[0]


def face_similarity(app, image_bgr: np.ndarray, ref_embed: np.ndarray) -> float:
    """Cosine similarity лучшего лица на изображении с эмбеддингом клиента.
    Если лицо не нашлось — возвращает -1."""
    faces = app.get(image_bgr)
    if not faces:
        return -1.0
    return max(float(np.dot(f.normed_embedding, ref_embed)) for f in faces)


# ----------------------------- helpers ---------------------------------------

def mask_crop_box(mask: np.ndarray, margin: float, img_w: int, img_h: int):
    """Квадратный кроп вокруг маски с запасом контекста."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Маска пустая.")
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w, h = x2 - x1, y2 - y1
    side = int(round(max(w, h) * (1.0 + 2.0 * margin)))
    side = min(side, img_w, img_h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nx1 = int(np.clip(cx - side // 2, 0, img_w - side))
    ny1 = int(np.clip(cy - side // 2, 0, img_h - side))
    return nx1, ny1, nx1 + side, ny1 + side


def dilate_px(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, kernel)


# ----------------------------- pipeline --------------------------------------

def build_pipeline(device: str, cfg: InpaintConfig):
    """SDXL-inpaint + ControlNet Tile + IP-Adapter FaceID (v1)."""
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Загрузка ControlNet Tile: {CONTROLNET_TILE_ID}")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_TILE_ID,
                                                 torch_dtype=dtype)

    print(f"Загрузка SDXL inpaint: {SDXL_INPAINT_ID}")
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        SDXL_INPAINT_ID, controlnet=controlnet, torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )

    print(f"Загрузка IP-Adapter FaceID: {FACEID_REPO}/{FACEID_WEIGHTS}")
    pipe.load_ip_adapter(FACEID_REPO, subfolder=None,
                         weight_name=FACEID_WEIGHTS, image_encoder_folder=None)

    adapters, weights = [], []
    pipe.load_lora_weights(FACEID_REPO, weight_name=FACEID_LORA,
                           adapter_name="faceid")
    adapters.append("faceid"); weights.append(cfg.faceid_lora_scale)

    if cfg.lora_path:
        print(f"Загрузка style LoRA: {cfg.lora_path} (scale={cfg.lora_scale})")
        pipe.load_lora_weights(cfg.lora_path, adapter_name="style")
        adapters.append("style"); weights.append(cfg.lora_scale)

    pipe.set_adapters(adapters, adapter_weights=weights)

    if device == "cuda":
        pipe.enable_model_cpu_offload()   # критично для 8GB VRAM
        pipe.enable_vae_tiling()
    else:
        pipe.to("cpu")
    return pipe


def faceid_embeds_for_pipe(embed: np.ndarray, device: str, dtype) -> torch.Tensor:
    """ArcFace-эмбеддинг -> формат ip_adapter_image_embeds."""
    pos = torch.from_numpy(embed).reshape(1, 1, -1)
    neg = torch.zeros_like(pos)
    return torch.cat([neg, pos], dim=0).to(device=device, dtype=dtype)


# ----------------------------- visualization ---------------------------------

def make_grid(candidates: list[dict], crop_box, out_path: Path) -> None:
    """Создание grid.jpg."""
    x1, y1, x2, y2 = crop_box
    tiles = []
    best = max(candidates, key=lambda d: d["similarity"])
    for c in candidates:
        tile = cv2.resize(c["image_bgr"][y1:y2, x1:x2], (512, 512))
        label = f"seed={c['seed']} sim={c['similarity']:.3f}"
        color = (0, 220, 0) if c is best else (255, 255, 255)
        cv2.rectangle(tile, (0, 0), (511, 511), color, 4 if c is best else 1)
        cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4)
        cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        tiles.append(tile)
    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    blank = np.full((512, 512, 3), 30, np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(out_path), grid)


# ----------------------------------- main ------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Вариант 3: коллаж волос клиента + SDXL inpaint "
                    "(IP-Adapter FaceID v1 + ControlNet Tile).")
    parser.add_argument("--image", required=True, help="Иллюстрация.")
    parser.add_argument("--client", required=True, help="Фото клиента.")
    parser.add_argument("--hair", default="",
                        help="Описание причёски (англ.), напр. 'long wavy "
                             "blonde hair'. Опционально, но помогает.")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--ip-scale", type=float, default=None)
    parser.add_argument("--faceid-lora-scale", type=float, default=None,
                        help="Вес вспомогательной LoRA FaceID (0.5..1.0).")
    parser.add_argument("--control-scale", type=float, default=None,
                        help="Вес ControlNet Tile (0.35..0.55).")
    parser.add_argument("--gen-size", type=int, default=None,
                        help="Рабочее разрешение (1024; 768 при OOM на 8GB).")
    parser.add_argument("--paste-face", action="store_true",
                        help="Вклеивать в коллаж и лицо клиента (эксперимент).")
    parser.add_argument("--lora", default=None, help="Путь к style LoRA.")
    parser.add_argument("--lora-scale", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.seeds is not None:
        overrides["seeds"] = args.seeds
    if args.ip_scale is not None:
        overrides["ip_scale"] = args.ip_scale
    if args.faceid_lora_scale is not None:
        overrides["faceid_lora_scale"] = args.faceid_lora_scale
    if args.control_scale is not None:
        overrides["control_scale"] = args.control_scale
    if args.gen_size is not None:
        overrides["gen_size"] = args.gen_size
    if args.lora is not None:
        overrides["lora_path"] = args.lora
    if args.lora_scale is not None:
        overrides["lora_scale"] = args.lora_scale
    if args.hair.strip():
        overrides["prompt"] = f"{CFG.prompt}, {args.hair.strip()}"
    cfg = dataclasses.replace(CFG, **overrides)

    image_path = Path(args.image)
    client_path = Path(args.client)
    stem = image_path.stem
    mask_path = MASKS_DIR / f"{stem}_mask.png"
    for p, what in [(image_path, "иллюстрация"), (client_path, "фото клиента"),
                    (mask_path, "маска")]:
        if not p.exists():
            raise FileNotFoundError(f"Не найден файл ({what}): {p}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # --- identity клиента ---
    print("Загрузка InsightFace.")
    face_app = build_face_analyzer()
    client_bgr, cface = client_face(face_app, client_path)
    ref_embed = cface.normed_embedding.astype(np.float32)
    print(f"Эмбеддинг клиента получен ({client_path.name}).")

    # --- иллюстрация + маска ---
    illus_bgr = cv2.imread(str(image_path))
    mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if illus_bgr is None or mask_full is None:
        raise ValueError("Не удалось прочитать иллюстрацию или маску.")
    if mask_full.shape[:2] != illus_bgr.shape[:2]:
        raise ValueError("Размеры маски и иллюстрации не совпадают.")
    img_h, img_w = illus_bgr.shape[:2]

    # ============================ ЭТАП 1: КОЛЛАЖ ============================
    print("Этап 1: сегментация волос клиента и коллаж...")
    col = make_collage_for(illus_bgr, mask_full, client_bgr, face_app, cface,
                           device, paste_face=args.paste_face)
    print(f"Коллаж готов (выравнивание: {col.align_method}, "
          f"волосы вклеены: {col.hair_pasted}, "
          f"лицо-персонажа сохранено: {col.face_preserved}, "
          f"метод затирания волос: {col.wipe_method}).")

    # маска инпейнта = регион головы (лицо+волосы+волосы клиента).
    # Вариант B: живописное лицо персонажа под маской СОХРАНЕНО в коллаже и
    # уходит в ControlNet Tile как texture-reference — поэтому перерисованное
    # лицо остаётся в мазках, а не превращается в гладкий "пластик".
    inpaint_full = col.inpaint_region.copy()

    # кроп считаем по этой маске: длинные волосы клиента должны попасть в кроп
    x1, y1, x2, y2 = mask_crop_box(inpaint_full, CROP_MARGIN, img_w, img_h)
    side = x2 - x1
    print(f"Кроп вокруг зоны: [{x1},{y1},{x2},{y2}] ({side}px) -> {cfg.gen_size}px")

    # запас на швы (задан в px кропа gen_size, пересчёт в масштаб кропа)
    seam = max(1, round(cfg.seam_dilate * side / cfg.gen_size))
    inpaint_full = dilate_px(inpaint_full, seam)

    crop_mask = inpaint_full[y1:y2, x1:x2]
    crop_collage = col.collage_bgr[y1:y2, x1:x2]

    gs = cfg.gen_size
    crop_rgb = cv2.resize(cv2.cvtColor(crop_collage, cv2.COLOR_BGR2RGB),
                          (gs, gs), interpolation=cv2.INTER_LANCZOS4)
    mask_gs = cv2.resize(crop_mask, (gs, gs), interpolation=cv2.INTER_NEAREST)
    if cfg.mask_blur > 0:
        k = cfg.mask_blur * 2 + 1
        mask_gs = cv2.GaussianBlur(mask_gs, (k, k), 0)

    pil_image = Image.fromarray(crop_rgb)     # база инпейнта — коллаж
    pil_control = Image.fromarray(crop_rgb)   # control image (Tile) — тоже коллаж
    pil_mask = Image.fromarray(mask_gs)

    # ====================== ЭТАП 2: ИНПЕЙНТ + CONTROLNET =====================
    pipe = build_pipeline(device, cfg)
    pipe.set_ip_adapter_scale(cfg.ip_scale)
    id_embeds = faceid_embeds_for_pipe(ref_embed, device, dtype)

    out_dir = OUTPUTS_DIR / f"{stem}__{client_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "collage.png"), col.collage_bgr)
    cv2.imwrite(str(out_dir / "inpaint_mask.png"), inpaint_full)

    # мягкая альфа для финальной вклейки (композим на ОРИГИНАЛ иллюстрации)
    alpha_full = cv2.GaussianBlur(crop_mask, (2 * cfg.mask_blur + 1,) * 2, 0)
    alpha_full = (alpha_full.astype(np.float32) / 255.0)[..., None]

    candidates = []
    for i in range(cfg.seeds):
        seed = cfg.base_seed + i
        gen = torch.Generator(device="cpu").manual_seed(seed)
        print(f"\nГенерация {i + 1}/{cfg.seeds} (seed={seed}) ...")
        result = pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative,
            image=pil_image,
            mask_image=pil_mask,
            control_image=pil_control,
            controlnet_conditioning_scale=cfg.control_scale,
            ip_adapter_image_embeds=[id_embeds],
            strength=cfg.strength,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            generator=gen,
            height=gs, width=gs,
        ).images[0]

        # кроп gen_size -> обратно в размер кропа -> мягкая вклейка по маске
        gen_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.resize(gen_bgr, (side, side), interpolation=cv2.INTER_LANCZOS4)
        composed = illus_bgr.copy()
        region = composed[y1:y2, x1:x2].astype(np.float32)
        blended = alpha_full * gen_bgr.astype(np.float32) + (1 - alpha_full) * region
        composed[y1:y2, x1:x2] = blended.astype(np.uint8)

        sim = face_similarity(face_app, composed[y1:y2, x1:x2], ref_embed)
        print(f"  similarity={sim:.2f}" + ("  (лицо не найдено!)" if sim < 0 else ""))

        cand_path = out_dir / f"candidate_{seed}.png"
        cv2.imwrite(str(cand_path), composed)
        candidates.append({"seed": seed, "similarity": sim,
                           "image_bgr": composed, "path": str(cand_path)})

    best = max(candidates, key=lambda c: c["similarity"])
    best_path = out_dir / "best.png"
    cv2.imwrite(str(best_path), best["image_bgr"])

    # quality gate: если лучший кандидат без детектируемого лица — это симптом
    # "лицо слилось с фоном". Явно сигналим и подсказываем, что крутить.
    if best["similarity"] < 0.15:
        print("\n[!] ВНИМАНИЕ: на лучшем кандидате лицо почти/совсем не "
              "детектируется (similarity низкий). Вероятно лицо слилось с "
              "фоном/потеряло структуру. Попробуйте усилить якорь лица:\n"
              "    --control-scale 0.6  (Tile сильнее держит структуру лица)\n"
              "    --ip-scale 1.0       (сильнее переносит личность клиента)\n"
              "    и проверьте wipe_method в логе: 'geometric' — самый грубый, "
              "возможно, маска головы (illustrations_mask) неточная.")

    grid_path = out_dir / "grid.jpg"
    make_grid(candidates, (x1, y1, x2, y2), grid_path)

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "variant": "v3_collage_controlnet",
            "image": str(image_path), "client": str(client_path),
            "mask": str(mask_path), "crop_box": [x1, y1, x2, y2],
            "faceid": FACEID_WEIGHTS, "controlnet": CONTROLNET_TILE_ID,
            "align_method": col.align_method, "hair_pasted": col.hair_pasted,
            "face_preserved": col.face_preserved, "wipe_method": col.wipe_method,
            "paste_face": bool(args.paste_face), "hair": args.hair,
            "prompt": cfg.prompt, "negative": cfg.negative,
            "ip_scale": cfg.ip_scale, "control_scale": cfg.control_scale,
            "strength": cfg.strength, "steps": cfg.steps,
            "guidance": cfg.guidance, "gen_size": cfg.gen_size,
            "seam_dilate": cfg.seam_dilate, "mask_blur": cfg.mask_blur,
            "lora": cfg.lora_path, "lora_scale": cfg.lora_scale,
            "candidates": [{"seed": c["seed"], "similarity": c["similarity"],
                            "path": c["path"]} for c in candidates],
            "best_seed": best["seed"], "best_similarity": best["similarity"],
            "best_path": str(best_path),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nЛучший seed: {best['seed']} (similarity={best['similarity']:.2f})")
    print(f"Итог:   {best_path}")
    print(f"Коллаж: {out_dir / 'collage.png'}")
    print(f"Сетка:  {grid_path}")
    print(f"JSON:   {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()