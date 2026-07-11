"""
В этом файле находится этап 3 пайплайна.

Как работает:
1) Identity-embedding клиента (InsightFace: ArcFace-эмбеддинг + выровненный кроп лица)
2) Stylized inpainting: SDXL-inpaint + IP-Adapter FaceID **PlusV2**
   (несколько seed, по ArcFace-сходству выдаётся лучший)
3) Результат сохраняется в виде
   best.png            - итог с максимальным score
   candidate_<seed>.png - все кандидаты
   grid.jpg            - сетка кандидатов
   result.json         - параметры

Почему PlusV2, а не обычный FaceID:
  обычный FaceID видит ТОЛЬКО ArcFace-эмбеддинг (геометрия лица), в нём нет
  информации о причёске/цвете волос — поэтому волосы "придумывались".
  PlusV2 добавляет CLIP-эмбеддинг кропа головы клиента: причёска, цвет волос
  и общий вид переносятся из реального фото. Критично для девушек с длинными
  волосами.

Важно: инпейнтим НЕ весь разворот, а квадрат вокруг маски, приведённый к 1024x1024.
Без кропа SDXL выдаст не похожее на данного человека изображение.
После генерации квадрат возвращается на место, заменяются только пиксели под маской.

Маска дополнительно расширяется (mask_dilate): если у персонажа на заготовке
волосы короче, чем у клиента, без расширения новым волосам некуда "вырасти".

Запуск:
  python output_inpainting.py --image data/illustrations/spread_01.png \
      --client data/clients/ivan.jpg \
      --hair "long wavy blonde hair"

  --hair — необязательное текстовое описание причёски клиента (на английском),
  сильно помогает в дополнение к CLIP-ветке PlusV2.

!!! Требует, чтобы для картинки уже была посчитана маска (illustrations_mask.py).
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

from diffusers import AutoPipelineForInpainting

from illustrations_mask import (
    PROJECT_ROOT, ILLUSTRATIONS_DIR, MASKS_DIR
)

OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
FACEID_REPO = "h94/IP-Adapter-FaceID"
# PlusV2: ArcFace-эмбеддинг + CLIP-эмбеддинг лица (переносит причёску/цвет волос)
FACEID_WEIGHTS = "ip-adapter-faceid-plusv2_sdxl.bin"
FACEID_LORA = "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
CLIP_IMAGE_ENCODER_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

GEN_SIZE = 512          # рабочее разрешение инпейнта (кроп -> 1024 -> обратно)
CROP_MARGIN = 0.4        # запас контекста вокруг маски (доля от размера маски)

DEFAULT_PROMPT = (
    "illustration, painted style, portrait of a person, "
    "same hairstyle and hair color as the reference person, "
    "accurate likeness of the reference face, "
    "brush strokes and palette consistent with the artwork, "
    "seamlessly integrated into the scene"
)
DEFAULT_NEGATIVE = (
    "photo, photorealistic, 3d render, deformed, blurry, low quality, "
    "extra eyes, bad anatomy, text, watermark, "
    "different person, generic face, altered facial features, "
    "different hairstyle, different hair color"
)


# ----------------------------- настройки инпейнта ----------------------------


@dataclass(frozen=True)
class InpaintConfig:
    lora_path: str | None = None    # путь к style LoRA
    lora_scale: float = 0.5
    ip_scale: float = 1.1           # сила identity (IP-Adapter), 0.8..1.4
    strength: float = 0.99          # сила инпейнта; 1.0 = полностью перерисовать зону
    steps: int = 35
    guidance: float = 5.0           # высокий CFG "тянет" к generic-лицу, 4.5..6
    seeds: int = 1                 # сколько кандидатов сгенерировать
    base_seed: int = 42
    mask_blur: int = 8              # размытие краёв маски (px в кропе 1024)
    mask_dilate: int = 16           # расширение маски (px в кропе 1024), запас под причёску
    prompt: str = DEFAULT_PROMPT
    negative: str = DEFAULT_NEGATIVE


CFG = InpaintConfig()


# ----------------------------- identity-embedding -----------------------------

def build_face_analyzer():
    """InsightFace buffalo_l: детекция + ArcFace-эмбеддинг """
    from insightface.app import FaceAnalysis
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available() else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def client_identity(app, photo_path: Path) -> tuple[np.ndarray, Image.Image]:
    """ArcFace-эмбеддинг клиента + выровненный кроп лица (для CLIP-ветки PlusV2).

    Кроп берём с запасом вокруг стандартного выравнивания, чтобы в кадр
    попали волосы целиком (это то, что "видит" CLIP-ветка).
    """
    from insightface.utils import face_align

    img = cv2.imread(str(photo_path))
    if img is None:
        raise ValueError(f"Не удалось прочитать фото клиента: {photo_path}")
    faces = app.get(img)
    if not faces:
        raise ValueError(f"На фото клиента не найдено лицо: {photo_path}.")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    face = faces[0]

    # стандартный выровненный кроп 224x224 (как в официальном демо FaceID Plus)
    aligned = face_align.norm_crop(img, landmark=face.kps, image_size=224)
    aligned_pil = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

    return face.normed_embedding.astype(np.float32), aligned_pil


def face_similarity(app, image_bgr: np.ndarray, ref_embed: np.ndarray) -> float:
    """Cosine similarity лучшего лица на изображении с эмбеддингом клиента.
    Если лицо не нашлось - возвращает -1."""
    faces = app.get(image_bgr)
    if not faces:
        return -1.0
    sims = [float(np.dot(f.normed_embedding, ref_embed)) for f in faces]
    return max(sims)


# ----------------------------- helpers ------------------------------

def mask_crop_box(mask: np.ndarray, margin: float, img_w: int, img_h: int):
    """Квадратный кроп вокруг маски с запасом контекста """
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


def dilate_mask(mask: np.ndarray, dilate_px_1024: int, side: int) -> np.ndarray:
    """Расширяет маску, чтобы у причёски клиента был запас места.
    dilate_px_1024 задан в пикселях кропа 1024, пересчитывается в масштаб кропа."""
    px = max(0, round(dilate_px_1024 * side / GEN_SIZE))
    if px == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, kernel)


# ----------------------------- stylized inpainting  ------------------------------

def build_pipeline(device: str, lora_path: str | None, lora_scale: float):
    """ SDXL-inpaint + IP-Adapter FaceID PlusV2 (+ его вспомогательная LoRA).

    PlusV2 требует CLIP image encoder: через него в модель попадает
    внешний вид клиента (причёска, цвет волос), а не только геометрия лица.
    """
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Загрузка CLIP image encoder: {CLIP_IMAGE_ENCODER_ID}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        CLIP_IMAGE_ENCODER_ID, torch_dtype=dtype
    )

    print(f"Загрузка SDXL inpaint: {SDXL_INPAINT_ID} ")
    pipe = AutoPipelineForInpainting.from_pretrained(
        SDXL_INPAINT_ID, torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        image_encoder=image_encoder,
        feature_extractor=CLIPImageProcessor(),
    )

    print(f"Загрузка IP-Adapter FaceID PlusV2: {FACEID_REPO}/{FACEID_WEIGHTS} ")
    pipe.load_ip_adapter(
        FACEID_REPO, subfolder=None, weight_name=FACEID_WEIGHTS,
        image_encoder_folder=None,
    )

    adapters, weights = [], []
    # Вспомогательная LoRA самого FaceID PlusV2
    pipe.load_lora_weights(FACEID_REPO, weight_name=FACEID_LORA,
                           adapter_name="faceid")
    adapters.append("faceid"); weights.append(1.0)

    if lora_path:
        print(f"Загрузка style LoRA: {lora_path} (scale={lora_scale})")
        pipe.load_lora_weights(lora_path, adapter_name="style")
        adapters.append("style"); weights.append(lora_scale)

    pipe.set_adapters(adapters, adapter_weights=weights)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
    else:
        pipe.to("cpu")
    return pipe


def inject_clip_face_embeds(pipe, face_pil: Image.Image, device: str, dtype) -> None:
    """CLIP-эмбеддинг кропа лица клиента -> projection-слой FaceID PlusV2.

    Это официальный способ diffusers для FaceID Plus/PlusV2:
    clip_embeds кладутся напрямую в image_projection_layers[0],
    shortcut=True включает режим v2.
    """
    clip_embeds = pipe.prepare_ip_adapter_image_embeds(
        [face_pil], None, torch.device(device), 1, True
    )[0]
    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    proj.clip_embeds = clip_embeds.to(dtype=dtype)
    proj.shortcut = True  # True = PlusV2


def faceid_embeds_for_pipe(embed: np.ndarray, device: str, dtype) -> torch.Tensor:
    """ArcFace-эмбеддинг -> формат ip_adapter_image_embeds """
    pos = torch.from_numpy(embed).reshape(1, 1, -1)
    neg = torch.zeros_like(pos)
    return torch.cat([neg, pos], dim=0).to(device=device, dtype=dtype)


# ----------------------------- visualization --------------------------------

def make_grid(candidates: list[dict], crop_box, out_path: Path) -> None:
    """ создание grid.jpg """
    x1, y1, x2, y2 = crop_box
    tiles = []
    for c in candidates:
        tile = cv2.resize(c["image_bgr"][y1:y2, x1:x2], (512, 512))
        label = f"seed={c['seed']} sim={c['similarity']:.3f}"
        color = (0, 220, 0) if c is max(candidates, key=lambda d: d["similarity"]) \
            else (255, 255, 255)
        cv2.rectangle(tile, (0, 0), (511, 511), color, 4 if color[1] == 220 else 1)
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
        description="Client Identity embedding + SDXL inpainting "
                    "(IP-Adapter FaceID PlusV2: лицо + причёска клиента)."
    )
    parser.add_argument("--image", required=True, help="Иллюстрация.")
    parser.add_argument("--client", required=True, help="Фото клиента.")
    parser.add_argument("--hair", default="",
                        help="Описание причёски клиента на английском, напр. "
                             "'long wavy blonde hair'. Сильно улучшает перенос волос.")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Сколько кандидатов генерировать (по умолчанию из CFG).")
    parser.add_argument("--ip-scale", type=float, default=None,
                        help="Сила identity 0.8..1.4 (по умолчанию из CFG).")
    parser.add_argument("--lora", default=None, help="Путь к style LoRA.")
    parser.add_argument("--lora-scale", type=float, default=None)
    args = parser.parse_args()

    # переопределения из CLI поверх CFG
    overrides = {}
    if args.seeds is not None:
        overrides["seeds"] = args.seeds
    if args.ip_scale is not None:
        overrides["ip_scale"] = args.ip_scale
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

    # --- identity ---
    print("Загрузка InsightFace.")
    face_app = build_face_analyzer()
    ref_embed, face_pil = client_identity(face_app, client_path)
    print(f"Эмбеддинг клиента получен ({client_path.name}).")

    # --- подготовка изображения/маски/кропа ---
    illus_bgr = cv2.imread(str(image_path))
    mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if illus_bgr is None or mask_full is None:
        raise ValueError("Не удалось прочитать иллюстрацию или маску.")
    if mask_full.shape[:2] != illus_bgr.shape[:2]:
        raise ValueError("Размеры маски и иллюстрации не совпадают.")
    img_h, img_w = illus_bgr.shape[:2]

    x1, y1, x2, y2 = mask_crop_box(mask_full, CROP_MARGIN, img_w, img_h)
    crop_bgr = illus_bgr[y1:y2, x1:x2]
    side = x2 - x1
    print(f"Кроп вокруг маски: [{x1},{y1},{x2},{y2}] ({side}px) -> {GEN_SIZE}px")

    # расширяем маску: запас места под причёску клиента
    crop_mask = dilate_mask(mask_full[y1:y2, x1:x2], cfg.mask_dilate, side)

    crop_rgb_1024 = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                               (GEN_SIZE, GEN_SIZE), interpolation=cv2.INTER_LANCZOS4)
    mask_1024 = cv2.resize(crop_mask, (GEN_SIZE, GEN_SIZE),
                           interpolation=cv2.INTER_NEAREST)
    if cfg.mask_blur > 0:
        k = cfg.mask_blur * 2 + 1
        mask_1024 = cv2.GaussianBlur(mask_1024, (k, k), 0)
    pil_image = Image.fromarray(crop_rgb_1024)
    pil_mask = Image.fromarray(mask_1024)

    # --- pipeline ---
    pipe = build_pipeline(device, cfg.lora_path, cfg.lora_scale)
    pipe.set_ip_adapter_scale(cfg.ip_scale)
    # CLIP-ветка PlusV2: внешний вид клиента (причёска/цвет волос)
    inject_clip_face_embeds(pipe, face_pil, device, dtype)
    id_embeds = faceid_embeds_for_pipe(ref_embed, device, dtype)

    out_dir = OUTPUTS_DIR / f"{stem}__{client_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- генерация кандидатов + отбор ---
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
            ip_adapter_image_embeds=[id_embeds],
            strength=cfg.strength,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            generator=gen,
            height=GEN_SIZE, width=GEN_SIZE,
        ).images[0]

        # кроп 1024 -> обратно в размер кропа -> вклейка только по маске
        # (по расширенной маске, чтобы не отрезать "выросшие" волосы)
        gen_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.resize(gen_bgr, (side, side), interpolation=cv2.INTER_LANCZOS4)
        composed = illus_bgr.copy()
        region = composed[y1:y2, x1:x2]
        m = (crop_mask > 127)
        region[m] = gen_bgr[m]
        composed[y1:y2, x1:x2] = region

        sim = face_similarity(face_app, composed[y1:y2, x1:x2], ref_embed)
        print(f"  similarity={sim:.2f}" + ("  (лицо не найдено!)" if sim < 0 else ""))

        cand_path = out_dir / f"candidate_{seed}.png"
        cv2.imwrite(str(cand_path), composed)
        candidates.append({"seed": seed, "similarity": sim,
                           "image_bgr": composed, "path": str(cand_path)})

    best = max(candidates, key=lambda c: c["similarity"])
    best_path = out_dir / "best.png"
    cv2.imwrite(str(best_path), best["image_bgr"])

    grid_path = out_dir / "grid.jpg"
    make_grid(candidates, (x1, y1, x2, y2), grid_path)

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "image": str(image_path), "client": str(client_path),
            "mask": str(mask_path), "crop_box": [x1, y1, x2, y2],
            "faceid": FACEID_WEIGHTS,
            "prompt": cfg.prompt, "negative": cfg.negative,
            "hair": args.hair, "mask_dilate": cfg.mask_dilate,
            "ip_scale": cfg.ip_scale, "strength": cfg.strength,
            "steps": cfg.steps, "guidance": cfg.guidance,
            "lora": cfg.lora_path, "lora_scale": cfg.lora_scale,
            "candidates": [{"seed": c["seed"], "similarity": c["similarity"],
                            "path": c["path"]} for c in candidates],
            "best_seed": best["seed"], "best_similarity": best["similarity"],
            "best_path": str(best_path),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nЛучший seed: {best['seed']} (similarity={best['similarity']:.2f})")
    print(f"Итог:   {best_path}")
    print(f"Сетка:  {grid_path}")
    print(f"JSON:   {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()