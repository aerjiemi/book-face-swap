"""
Этап 3 пайплайна: коллаж -> SDXL inpaint -> заливка нимба.

Как работает:
1) Collage: волосы клиента вклеиваются на заготовку (hair_collage)
2) Inpaint: SDXL + ControlNet Tile + IP-Adapter FaceID PlusV2
3) Заливка "нимба" вокруг новой головы доп. SDXL-проходом (background_fill)

Запуск:
  python output_inpainting.py --image data/illustrations/spread_01.png \
      --client data/clients/ivan.jpg

Флаги:
  --seeds N           сколько кандидатов (default 5)
  --likeness          приоритет сходства с клиентом над стилем
  --ip-scale          сила identity FaceID (0.6..1.0)
  --control-scale     вес ControlNet Tile (0.3..0.5); больше = точнее форма
                      волос и пропорции, но фактура "фотографичнее"
  --faceid-lora-scale вес вспомогательной LoRA FaceID (0.5..1.0)
  --paste-dilate      запас вклейки в px; больше = мягче шов, но риск "нимба"
  --no-bg-fill        отключить заливку нимба
  --paste-face        вклеивать в коллаж и лицо клиента

!!! Требует предрасчитанной маски (illustrations_mask.py).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


@contextlib.contextmanager
def quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


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
from background_fill import BackgroundFiller

OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_TILE_ID = "xinsir/controlnet-tile-sdxl-1.0"

FACEID_REPO = "h94/IP-Adapter-FaceID"
FACEID_V1_WEIGHTS = "ip-adapter-faceid_sdxl.bin"
FACEID_V1_LORA = "ip-adapter-faceid_sdxl_lora.safetensors"
FACEID_V2_WEIGHTS = "ip-adapter-faceid-plusv2_sdxl.bin"
FACEID_V2_LORA = "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"

IMAGE_ENCODER_REPO = "h94/IP-Adapter"
IMAGE_ENCODER_SUBFOLDER = "models/image_encoder"

CROP_MARGIN = 0.4   # запас контекста вокруг зоны инпейнта (доля от её размера)

DEFAULT_PROMPT = (
    "continuous background seamlessly matching the original image, exact same art style and colors as surroundings, perfectly integrated scene, "
    "painted portrait of a person, in the exact art style of the artwork, "
    "visible brush strokes, illustration, highly detailed face and eyes, "
    "correct natural facial proportions, "
    "perfect accurate likeness of the reference face, "
    "hair shape and hair color exactly as in the image, "
    "seamlessly integrated into the scene"
)
DEFAULT_NEGATIVE = (
    "plain background, solid color background, studio background, empty space, halo, vignette, "
    "photo, photorealistic, photographic skin, photographic hair texture, "
    "smooth skin, airbrushed, plastic skin, 3d render, vector, flat shading, "
    "elongated face, long face, narrow face, distorted proportions, "
    "deformed, blurry, low quality, extra eyes, bad anatomy, cross-eyed, "
    "text, watermark, different person, generic face"
)

# промпт при флаге --likeness
LIKENESS_PROMPT = (
    "continuous background seamlessly matching the original image, "
    "same colors as surroundings, perfectly integrated scene, "
    "highly detailed realistic portrait of a person, "
    "perfect accurate likeness of the reference face, "
    "exact same facial features as the reference person, "
    "highly detailed face and eyes, correct natural facial proportions, "
    "hair shape and hair color exactly as in the image, "
    "seamlessly integrated into the scene"
)
LIKENESS_NEGATIVE = (
    "plain background, solid color background, studio background, empty space, halo, vignette, "
    "elongated face, long face, narrow face, distorted proportions, "
    "stylized face, cartoon face, generic face, doll face, chibi, "
    "deformed, blurry, low quality, extra eyes, bad anatomy, cross-eyed, "
    "text, watermark, different person"
)


# ----------------------------- настройки инпейнта ----------------------------

@dataclass(frozen=True)
class InpaintConfig:
    ip_scale: float = 0.75         # сила identity (FaceID), 0.6..1.0
    faceid_lora_scale: float = 0.9 # вес вспомогательной LoRA FaceID
    control_scale: float = 0.2     # вес ControlNet Tile, 0.3..0.5
    strength: float = 0.6          # 1.0 = полностью перерисовать зону
    steps: int = 35
    guidance: float = 4.5
    seeds: int = 5                 # сколько кандидатов сгенерировать
    base_seed: int = 42
    gen_size: int = 1024           # рабочее разрешение (SDXL обучен на 1024)
    mask_blur: int = 8             # размытие краёв масок (px в кропе gen_size)
    seam_dilate: int = 48          # запас маски ГЕНЕРАЦИИ на швы (px, gen_size)
    paste_dilate: int = 30         # запас маски ВКЛЕЙКИ (px, gen_size). Мало
                                   # = нет "нимба"; много = мягче шов, но ореол
    faceid_version: str = "plusv2" # "plusv2" | "v1"
    base_model: str | None = "SG161222/RealVisXL_V5.0"
    bg_fill: bool = True           # заливка нимба вокруг головы
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
    with quiet():
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=(1024, 1024), det_thresh=0.3)
    return app


def client_face(app, photo_path: Path):
    """Крупнейшее лицо на фото клиента.

    """
    from PIL import ImageOps

    try:
        img_pil = Image.open(photo_path)
        img_pil = ImageOps.exif_transpose(img_pil)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(str(photo_path))
    if img is None:
        raise ValueError(f"Не удалось прочитать фото клиента: {photo_path}")

    faces = app.get(img)

    # фолбэк 1: сверхкрупный план — добавляем поля вокруг кадра
    if not faces:
        h, w = img.shape[:2]
        pad_h, pad_w = int(h * 0.25), int(w * 0.25)
        img_padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
        faces = app.get(img_padded)
        if faces:
            img = img_padded  # чтобы ключевые точки совпали с картинкой
        else:
            # фолбэк 2: экстремальное уменьшение для детектора
            old_det_size = app.det_size
            with quiet():
                app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.2)
                faces = app.get(img)
                app.prepare(ctx_id=0, det_size=old_det_size, det_thresh=0.3)

    if not faces:
        raise ValueError(f"На фото клиента не найдено лицо: {photo_path}")

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
    """SDXL(-inpaint) + ControlNet Tile + IP-Adapter FaceID (v1 или PlusV2)."""
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
    from diffusers.utils import logging as diffusers_logging
    from transformers.utils import logging as transformers_logging

    diffusers_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()

    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_TILE_ID,
                                                 torch_dtype=dtype)

    extra = {}
    if cfg.faceid_version == "plusv2":
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        extra["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            IMAGE_ENCODER_REPO, subfolder=IMAGE_ENCODER_SUBFOLDER,
            torch_dtype=dtype)
        extra["feature_extractor"] = CLIPImageProcessor()

    base_id = cfg.base_model or SDXL_INPAINT_ID
    try:
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_id, controlnet=controlnet, torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None, **extra)
    except (OSError, ValueError, EnvironmentError):
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_id, controlnet=controlnet, torch_dtype=dtype, **extra)

    if cfg.faceid_version == "plusv2":
        weights, lora = FACEID_V2_WEIGHTS, FACEID_V2_LORA
    else:
        weights, lora = FACEID_V1_WEIGHTS, FACEID_V1_LORA

    pipe.load_ip_adapter(FACEID_REPO, subfolder=None,
                         weight_name=weights, image_encoder_folder=None)
    pipe.load_lora_weights(FACEID_REPO, weight_name=lora, adapter_name="faceid")
    pipe.set_adapters(["faceid"], adapter_weights=[cfg.faceid_lora_scale])

    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        if total_gb >= 18:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
    else:
        pipe.to("cpu")

    pipe.set_progress_bar_config(disable=True)
    return pipe


def faceid_embeds_for_pipe(embed: np.ndarray, device: str, dtype) -> torch.Tensor:
    """ArcFace-эмбеддинг -> формат ip_adapter_image_embeds (для v1 и PlusV2)."""
    pos = torch.from_numpy(embed).reshape(1, 1, -1)
    neg = torch.zeros_like(pos)
    return torch.cat([neg, pos], dim=0).to(device=device, dtype=dtype)


def setup_faceid_plusv2(pipe, client_bgr: np.ndarray, cface,
                        device: str, dtype) -> None:
    """CLIP-ветка FaceID Plus V2: эмбеддинг выровненного кропа лица клиента."""
    from insightface.utils import face_align

    crop_bgr = face_align.norm_crop(
        client_bgr, landmark=cface.kps.astype(np.float32), image_size=224)
    pil_face = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    clip_embeds = pipe.prepare_ip_adapter_image_embeds(
        [pil_face], None, torch.device(device), 1, True)[0]

    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    proj.clip_embeds = clip_embeds.to(device=device, dtype=dtype)
    proj.shortcut = True  # True = именно Plus V2


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
        description="Коллаж волос клиента + SDXL inpaint (FaceID PlusV2 + "
                    "ControlNet Tile) + заливка нимба.")
    parser.add_argument("--image", required=True, help="Иллюстрация.")
    parser.add_argument("--client", required=True, help="Фото клиента.")
    parser.add_argument("--likeness", action="store_true",
                        help="Приоритет сходства с клиентом над стилем: "
                             "strength=1.0, ip_scale=1.0, control_scale вниз, "
                             "промпт разрешает реализм.")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--ip-scale", type=float, default=None)
    parser.add_argument("--faceid-lora-scale", type=float, default=None,
                        help="Вес вспомогательной LoRA FaceID (0.5..1.0).")
    parser.add_argument("--control-scale", type=float, default=None,
                        help="Вес ControlNet Tile (0.3..0.5).")
    parser.add_argument("--paste-dilate", type=int, default=None,
                        help="Запас маски ВКЛЕЙКИ в px кропа gen_size. "
                             "Большие значения возвращают 'нимб'.")
    parser.add_argument("--faceid", default=None, choices=["v1", "plusv2"],
                        help="Версия IP-Adapter FaceID (default: plusv2).")
    parser.add_argument("--base-model", default=None,
                        help="Другой SDXL-чекпоинт (HF id или путь).")
    parser.add_argument("--no-bg-fill", action="store_true",
                        help="Отключить заливку 'нимба' вокруг головы.")
    parser.add_argument("--paste-face", action="store_true",
                        help="Вклеивать в коллаж и лицо клиента (эксперимент).")
    args = parser.parse_args()

    overrides = {}
    mapping = {"seeds": args.seeds, "ip_scale": args.ip_scale,
               "faceid_lora_scale": args.faceid_lora_scale,
               "control_scale": args.control_scale,
               "paste_dilate": args.paste_dilate,
               "faceid_version": args.faceid, "base_model": args.base_model}
    for k, v in mapping.items():
        if v is not None:
            overrides[k] = v
    if args.no_bg_fill:
        overrides["bg_fill"] = False

    # режим приоритета сходства
    if args.likeness:
        likeness_preset = {
            "prompt": LIKENESS_PROMPT,
            "negative": LIKENESS_NEGATIVE,
            "control_scale": 0.18,   # Tile слабо держит форму персонажа
            "strength": 1.0,         # полностью перерисовать зону
            "ip_scale": 1.0,         # максимум переноса личности клиента
            "guidance": 5.0,
        }
        for k, v in likeness_preset.items():
            overrides.setdefault(k, v)

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

    print(f"Сцена: {image_path.name} | Клиент: {client_path.name}"
          + ("  [likeness]" if args.likeness else ""))

    # --- identity клиента ---
    face_app = build_face_analyzer()
    client_bgr, cface = client_face(face_app, client_path)
    ref_embed = cface.normed_embedding.astype(np.float32)

    # --- иллюстрация + маска ---
    illus_bgr = cv2.imread(str(image_path))
    mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if illus_bgr is None or mask_full is None:
        raise ValueError("Не удалось прочитать иллюстрацию или маску.")
    if mask_full.shape[:2] != illus_bgr.shape[:2]:
        raise ValueError("Размеры маски и иллюстрации не совпадают.")
    img_h, img_w = illus_bgr.shape[:2]

    # ============================ ЭТАП 1: КОЛЛАЖ ============================
    print("[1/3] Коллаж волос клиента ...")
    with quiet():
        col = make_collage_for(illus_bgr, mask_full, client_bgr, face_app,
                               cface, device, paste_face=args.paste_face)
    print(f"      выравнивание={col.align_method}, "
          f"волосы={'да' if col.hair_pasted else 'нет'}, "
          f"затирание={col.wipe_method}")

    inpaint_region = col.inpaint_region.copy()
    x1, y1, x2, y2 = mask_crop_box(inpaint_region, CROP_MARGIN, img_w, img_h)
    side = x2 - x1

    # ДВЕ маски: широкая для генерации (запас на швы), узкая для вклейки —
    # кольцо перегенерированного фона между ними отбрасывается
    gen_seam = max(1, round(cfg.seam_dilate * side / cfg.gen_size))
    paste_seam = max(0, round(cfg.paste_dilate * side / cfg.gen_size))
    gen_mask_full = dilate_px(inpaint_region, gen_seam)
    paste_mask_full = dilate_px(inpaint_region, paste_seam)

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

    # ====================== ЭТАП 2: ИНПЕЙНТ + CONTROLNET =====================
    print("[2/3] Загрузка моделей (SDXL + ControlNet + FaceID) ...")
    pipe = build_pipeline(device, cfg)
    pipe.set_ip_adapter_scale(cfg.ip_scale)
    if cfg.faceid_version == "plusv2":
        setup_faceid_plusv2(pipe, client_bgr, cface, device, dtype)
    id_embeds = faceid_embeds_for_pipe(ref_embed, device, dtype)

    bg_filler = None
    if cfg.bg_fill:
        bg_filler = BackgroundFiller(device, pipe=pipe, gen_size=cfg.gen_size,
                                     control_scale=cfg.control_scale,
                                     restore_ip_scale=cfg.ip_scale)
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

    # ========================= ЭТАП 3: ГЕНЕРАЦИЯ =============================
    print(f"[3/3] Генерация {cfg.seeds} кандидатов ...")
    candidates = []
    for i in range(cfg.seeds):
        seed = cfg.base_seed + i
        gen = torch.Generator(device="cpu").manual_seed(seed)
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

        # gen_size -> размер кропа -> мягкая вклейка по узкой маске на оригинал
        gen_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.resize(gen_bgr, (side, side), interpolation=cv2.INTER_LANCZOS4)
        composed = illus_bgr.copy()
        region = composed[y1:y2, x1:x2].astype(np.float32)
        blended = alpha_full * gen_bgr.astype(np.float32) + (1 - alpha_full) * region
        composed[y1:y2, x1:x2] = blended.astype(np.uint8)

        # заливка нимба
        bg_status, halo_mask = "disabled", None
        if bg_filler is not None:
            with quiet():
                composed, halo_mask, bg_status = bg_filler(
                    composed, paste_mask_full, inpaint_region)

        sim = face_similarity(face_app, composed[y1:y2, x1:x2], ref_embed)
        print(f"      {i + 1}/{cfg.seeds}  seed={seed}  similarity={sim:.2f}"
              f"  нимб={bg_status}"
              + ("  [лицо не найдено!]" if sim < 0 else ""))

        cand_path = out_dir / f"candidate_{seed}.png"
        cv2.imwrite(str(cand_path), composed)
        candidates.append({"seed": seed, "similarity": sim,
                           "image_bgr": composed, "path": str(cand_path),
                           "bg_status": bg_status, "halo_mask": halo_mask})

    best = max(candidates, key=lambda c: c["similarity"])
    best_path = out_dir / "best.png"
    cv2.imwrite(str(best_path), best["image_bgr"])
    if best.get("halo_mask") is not None:
        cv2.imwrite(str(out_dir / "bg_fill_mask.png"), best["halo_mask"])

    # quality gate: лицо слилось с фоном / потеряло структуру
    if best["similarity"] < 0.15:
        print("\n[!] ВНИМАНИЕ: на лучшем кандидате лицо почти не детектируется. "
              "Попробуйте --control-scale 0.5 и/или --ip-scale 1.0; "
              "wipe_method='geometric' в логе — признак неточной маски головы.")

    grid_path = out_dir / "grid.jpg"
    make_grid(candidates, (x1, y1, x2, y2), grid_path)

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "variant": "v3.3_collage_controlnet_plusv2_sdxl_bgfill",
            "image": str(image_path), "client": str(client_path),
            "mask": str(mask_path), "crop_box": [x1, y1, x2, y2],
            "faceid_version": cfg.faceid_version,
            "base_model": cfg.base_model or SDXL_INPAINT_ID,
            "controlnet": CONTROLNET_TILE_ID,
            "likeness": bool(args.likeness),
            "align_method": col.align_method, "hair_pasted": col.hair_pasted,
            "face_preserved": col.face_preserved, "wipe_method": col.wipe_method,
            "paste_face": bool(args.paste_face),
            "bg_fill": cfg.bg_fill,
            "prompt": cfg.prompt, "negative": cfg.negative,
            "ip_scale": cfg.ip_scale,
            "faceid_lora_scale": cfg.faceid_lora_scale,
            "control_scale": cfg.control_scale,
            "strength": cfg.strength, "steps": cfg.steps,
            "guidance": cfg.guidance, "gen_size": cfg.gen_size,
            "seam_dilate": cfg.seam_dilate, "paste_dilate": cfg.paste_dilate,
            "mask_blur": cfg.mask_blur,
            "candidates": [{"seed": c["seed"], "similarity": c["similarity"],
                            "path": c["path"], "bg_fill": c["bg_status"]}
                           for c in candidates],
            "best_seed": best["seed"], "best_similarity": best["similarity"],
            "best_path": str(best_path),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nЛучший seed: {best['seed']} (similarity={best['similarity']:.2f})")
    print(f"Итог:  {best_path}")
    print(f"Сетка: {grid_path}")


if __name__ == "__main__":
    main()