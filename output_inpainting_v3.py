"""
Этап 3 пайплайна, ВАРИАНТ 3.2: двухэтапный пайплайн (коллаж -> инпейнт + ControlNet)
+ финальная заливка "нимба" фоном (background_fill, LaMa).

Изменения относительно v3.1:

7) ЭТАП 3 — фикс "нимба-пустоты" внутри старого контура персонажа.
   Прежний фикс (п.3, разведение gen/paste масок) убирал ореол СНАРУЖИ
   старого контура. Но если голова клиента получается МЕНЬШЕ головы
   исходного персонажа, внутри старого контура остаётся кольцо
   перегенерированного псевдо-фона — оно лежит ВНУТРИ paste-маски и
   параметрами seam/paste_dilate не лечится. Теперь после генерации:
     - face-parsing сегментирует НОВУЮ голову на готовом кадре
       (защищаются лицо, волосы, уши, ШЕЯ и одежда — не изменяются);
     - нимб = paste-регион минус защищённая зона;
     - нимб заливается LaMa (big-lama) — моделью заполнения фона,
       которая продолжает мазки окружения без промпта.
   Отключить: --no-bg-fill. Требует: pip install simple-lama-inpainting.

Изменения относительно v3 (по итогам тестов на 3090/4090):

1) IP-Adapter FaceID -> FaceID Plus V2 (ip-adapter-faceid-plusv2_sdxl).
   Кроме ArcFace-эмбеддинга в модель подаётся CLIP-эмбеддинг выровненного
   кропа лица клиента. Это главное лекарство от "слишком детское /
   вытянутое / не похоже": ArcFace кодирует identity, но плохо переносит
   возраст, пол и пропорции головы — CLIP-ветка Plus V2 это добирает.
   Откат на старое поведение: --faceid v1.

2) Авто-описание клиента в промпте: InsightFace (buffalo_l) отдаёт пол и
   примерный возраст -> в промпт подставляется "35 year old man" и т.п.,
   для взрослых в negative добавляется "child, baby face, ...".
   Переопределить: --person "40 year old bearded man"; отключить: --person "".

3) Фикс "нимба" вокруг головы (внешнего). Раньше маска ГЕНЕРАЦИИ
   (расширенная на seam_dilate) использовалась и для финальной ВКЛЕЙКИ —
   кольцо перегенерированного фона (чуть другого цвета) вклеивалось поверх
   оригинала и было видно как ореол. Теперь маски разведены:
     - генерация: регион + seam_dilate (модели нужен запас на швы);
     - вклейка:   регион + paste_dilate (по умолчанию 6px, почти впритык).
   Фон вокруг головы в итоговой картинке — всегда оригинальные пиксели.

4) Промпт: убран конфликт "clear smooth skin" (positive) vs "smooth skin"
   (negative), в negative добавлены анти-"вытянутое лицо" токены.
   control_scale по умолчанию 0.28 -> 0.35: живописное лицо персонажа в
   коллаже через ControlNet Tile держит ПРОПОРЦИИ головы, а ЧЕРТЫ лица
   задаёт FaceID.

5) --base-model: другой SDXL-чекпоинт вместо diffusers/sdxl-inpainting
   (например RunDiffusion/Juggernaut-XL-v9 или SG161222/RealVisXL_V5.0).
   Pipeline умеет инпейнтить и обычным (не-inpaint) UNet. ВНИМАНИЕ:
   фотореалистичные чекпоинты рисуют лица структурно лучше, но тянут
   в фотореализм против иллюстративного стиля — проверять глазами.

6) На GPU с >=18GB VRAM (3090/4090) пайплайн грузится целиком на CUDA
   (быстро); cpu-offload включается автоматически только на малых картах.

Запуск:
  python output_inpainting_v3.py --image data/illustrations/spread_01.png \
      --client data/clients/ivan.jpg --hair "long wavy blonde hair"

  # сравнить со старым FaceID v1:
  python output_inpainting_v3.py ... --faceid v1

  # попробовать другой базовый чекпоинт:
  python output_inpainting_v3.py ... --base-model RunDiffusion/Juggernaut-XL-v9

  --hair          текст про причёску (англ.) — опционально, но помогает;
  --person        описание клиента для промпта (по умолчанию авто по
                  возрасту/полу из InsightFace; --person "" отключает);
  --no-bg-fill    отключить этап 3 (заливку нимба фоном через LaMa);
  --control-scale вес ControlNet Tile (0.3..0.5); больше = точнее форма
                  волос и пропорции лица, но фактура "фотографичнее";
  --paste-dilate  запас вклейки в px (кроп gen_size); больше = мягче шов,
                  но возвращается риск "нимба";
  --gen-size 768  при OOM на 8GB (1024 по умолчанию — SDXL обучен на 1024).

!!! Требует предрасчитанной маски (illustrations_mask.py).
Первый запуск докачает: ControlNet Tile (~2.5GB), face-parsing (~340MB),
IP-Adapter FaceID PlusV2 (~1.3GB), CLIP image encoder ViT-H (~2.5GB),
LaMa big-lama (~200MB, для этапа 3).
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
from background_fill import BackgroundFiller

OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_TILE_ID = "xinsir/controlnet-tile-sdxl-1.0"

FACEID_REPO = "h94/IP-Adapter-FaceID"
FACEID_V1_WEIGHTS = "ip-adapter-faceid_sdxl.bin"
FACEID_V1_LORA = "ip-adapter-faceid_sdxl_lora.safetensors"
FACEID_V2_WEIGHTS = "ip-adapter-faceid-plusv2_sdxl.bin"
FACEID_V2_LORA = "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
# CLIP-энкодер для CLIP-ветки Plus V2 (ViT-H, тот же, что у обычного IP-Adapter)
IMAGE_ENCODER_REPO = "h94/IP-Adapter"
IMAGE_ENCODER_SUBFOLDER = "models/image_encoder"

CROP_MARGIN = 0.4   # запас контекста вокруг зоны инпейнта (доля от её размера)

# {person} подставляется в main (авто по возрасту/полу либо --person)
DEFAULT_PROMPT = (
    "continuous background seamlessly matching the original image, exact same art style and colors as surroundings, perfectly integrated scene, "
    "painted portrait of a {person}, in the exact art style of the artwork, "
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


# ----------------------------- настройки инпейнта ----------------------------

@dataclass(frozen=True)
class InpaintConfig:
    lora_path: str | None = None   # путь к style LoRA
    lora_scale: float = 0.7
    ip_scale: float = 1.0          # сила identity (FaceID), 0.6..1.0
    faceid_lora_scale: float = 0.9 # вес вспомогательной LoRA FaceID
    control_scale: float = 0.35  # вес ControlNet Tile, 0.3..0.5
    strength: float = 0.95         # 1.0 = полностью перерисовать зону
    steps: int = 35
    guidance: float = 4.5
    seeds: int = 5                 # сколько кандидатов сгенерировать
    base_seed: int = 42
    gen_size: int = 1024           # рабочее разрешение (SDXL обучен на 1024!)
    mask_blur: int = 8         # размытие краёв масок (px в кропе gen_size)
    seam_dilate: int = 64        # запас маски ГЕНЕРАЦИИ на швы (px, gen_size)
    paste_dilate: int = 2         # запас маски ВКЛЕЙКИ (px, gen_size). Мало
                                   # = нет "нимба"; много = мягче шов, но ореол
    faceid_version: str = "plusv2" # "plusv2" | "v1"
    base_model: str | None = "SG161222/RealVisXL_V5.0"  # другой SDXL-чекпоинт вместо inpaint
    bg_fill: bool = True           # этап 3: заливка нимба фоном (LaMa)
    prompt: str = DEFAULT_PROMPT
    negative: str = DEFAULT_NEGATIVE


CFG = InpaintConfig()


# ----------------------------- identity-embedding ----------------------------

def build_face_analyzer():
    """InsightFace buffalo_l: детекция + ArcFace-эмбеддинг + пол/возраст."""
    from insightface.app import FaceAnalysis
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available() else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(1024, 1024), det_thresh=0.3)
    return app


def client_face(app, photo_path: Path):
    """Крупнейшее лицо на фото клиента (с авто-фиксами поворота и сверхкрупного плана).

    Фото клиента по продукту всегда крупноплановое и качественное: обычная
    детекция на det_size=1024 срабатывает почти всегда; фолбэки ниже
    закрывают редкий случай ЭКСТРЕМАЛЬНО крупного плана (лицо на весь кадр).
    """
    from PIL import ImageOps

    # 1. Загрузка изображения с исправлением поворота EXIF (телефонные фото)
    try:
        img_pil = Image.open(photo_path)
        img_pil = ImageOps.exif_transpose(img_pil)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(str(photo_path))

    if img is None:
        raise ValueError(f"Не удалось прочитать фото клиента: {photo_path}")

    # 2. Первая попытка детекции (в лоб)
    faces = app.get(img)

    # 3. Если лицо НЕ найдено — гипотеза "сверхкрупного плана" (добавляем поля)
    if not faces:
        print("[ИНФО] Лицо не найдено. Возможно, оно слишком крупное. "
              "Искусственно расширяем границы...")
        h, w = img.shape[:2]
        pad_h, pad_w = int(h * 0.25), int(w * 0.25)
        img_padded = cv2.copyMakeBorder(
            img, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        faces = app.get(img_padded)
        if faces:
            print("[УСПЕХ] Лицо обнаружено после добавления полей!")
            img = img_padded  # чтобы ключевые точки совпали с картинкой
        else:
            print("[ИНФО] Поля не сработали. Пробуем экстремальное сжатие "
                  "для детектора...")
            old_det_size = app.det_size
            app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.2)
            faces = app.get(img)
            app.prepare(ctx_id=0, det_size=old_det_size, det_thresh=0.3)

    if not faces:
        raise ValueError(
            f"На фото клиента не найдено лицо (даже после авто-фиксов): {photo_path}."
        )

    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    return img, faces[0]


def person_descriptor(face) -> tuple[str, str]:
    """"35 year old man" по атрибутам InsightFace + доп. negative-токены.

    Возраст buffalo_l оценивает грубо (+-5..7 лет) — этого достаточно, чтобы
    оттащить SDXL от дефолтного "молодого generic-лица". Точное описание
    всегда можно задать через --person.
    """
    age = getattr(face, "age", None)
    sex = getattr(face, "sex", None)
    if age is None or sex is None:
        return "person", ""
    age = int(age)
    male = str(sex).upper().startswith("M")

    if age < 4:
        noun = "toddler boy" if male else "toddler girl"
        desc = noun
    elif age < 13:
        noun = "boy" if male else "girl"
        desc = f"{age} year old {noun}"
    elif age < 20:
        noun = "teenage boy" if male else "teenage girl"
        desc = f"{age} year old {noun}"
    else:
        noun = "man" if male else "woman"
        desc = f"{age} year old {noun}"

    if age >= 20:
        extra_neg = "child, kid, teenager, childlike face, baby face, chibi"
    elif age < 13:
        extra_neg = "adult, elderly, wrinkles, facial hair"
    else:
        extra_neg = ""
    return desc, extra_neg


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

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Загрузка ControlNet Tile: {CONTROLNET_TILE_ID}")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_TILE_ID,
                                                 torch_dtype=dtype)

    # для PlusV2 нужен CLIP image encoder (CLIP-ветка identity)
    extra = {}
    if cfg.faceid_version == "plusv2":
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        print(f"Загрузка CLIP image encoder: "
              f"{IMAGE_ENCODER_REPO}/{IMAGE_ENCODER_SUBFOLDER}")
        extra["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            IMAGE_ENCODER_REPO, subfolder=IMAGE_ENCODER_SUBFOLDER,
            torch_dtype=dtype)
        extra["feature_extractor"] = CLIPImageProcessor()

    base_id = cfg.base_model or SDXL_INPAINT_ID
    print(f"Загрузка базовой модели: {base_id}")
    try:
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_id, controlnet=controlnet, torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None, **extra)
    except (OSError, ValueError, EnvironmentError):
        # у многих кастомных чекпоинтов нет fp16-варианта весов
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_id, controlnet=controlnet, torch_dtype=dtype, **extra)

    if cfg.faceid_version == "plusv2":
        weights, lora = FACEID_V2_WEIGHTS, FACEID_V2_LORA
    else:
        weights, lora = FACEID_V1_WEIGHTS, FACEID_V1_LORA

    print(f"Загрузка IP-Adapter FaceID ({cfg.faceid_version}): "
          f"{FACEID_REPO}/{weights}")
    pipe.load_ip_adapter(FACEID_REPO, subfolder=None,
                         weight_name=weights, image_encoder_folder=None)

    adapters, adapter_weights = [], []
    pipe.load_lora_weights(FACEID_REPO, weight_name=lora, adapter_name="faceid")
    adapters.append("faceid"); adapter_weights.append(cfg.faceid_lora_scale)

    if cfg.lora_path:
        print(f"Загрузка style LoRA: {cfg.lora_path} (scale={cfg.lora_scale})")
        pipe.load_lora_weights(cfg.lora_path, adapter_name="style")
        adapters.append("style"); adapter_weights.append(cfg.lora_scale)

    pipe.set_adapters(adapters, adapter_weights=adapter_weights)

    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        if total_gb >= 18:          # 3090/4090 и крупнее — всё на GPU, быстро
            print(f"VRAM {total_gb:.0f}GB: пайплайн целиком на CUDA.")
            pipe.to("cuda")
        else:                        # 8-16GB — офлоад, медленно, но работает
            print(f"VRAM {total_gb:.0f}GB: включаю cpu-offload + vae tiling.")
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
    else:
        pipe.to("cpu")
    return pipe


def faceid_embeds_for_pipe(embed: np.ndarray, device: str, dtype) -> torch.Tensor:
    """ArcFace-эмбеддинг -> формат ip_adapter_image_embeds (для v1 и PlusV2)."""
    pos = torch.from_numpy(embed).reshape(1, 1, -1)
    neg = torch.zeros_like(pos)
    return torch.cat([neg, pos], dim=0).to(device=device, dtype=dtype)


def setup_faceid_plusv2(pipe, client_bgr: np.ndarray, cface,
                        device: str, dtype) -> None:
    """CLIP-ветка FaceID Plus V2: эмбеддинг выровненного кропа лица клиента.

    Кроп 224x224 делается стандартным ArcFace-выравниванием по 5 точкам
    (insightface.face_align), прогоняется через CLIP ViT-H и кладётся в
    projection layer адаптера (механика из официальной документации diffusers).
    """
    from insightface.utils import face_align

    crop_bgr = face_align.norm_crop(
        client_bgr, landmark=cface.kps.astype(np.float32), image_size=224)
    pil_face = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    clip_embeds = pipe.prepare_ip_adapter_image_embeds(
        [pil_face], None, torch.device(device), 1, True)[0]

    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    proj.clip_embeds = clip_embeds.to(device=device, dtype=dtype)
    proj.shortcut = True  # True = именно Plus V2
    print("FaceID Plus V2: CLIP-эмбеддинг лица клиента подготовлен.")


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
        description="Вариант 3.2: коллаж волос клиента + SDXL inpaint "
                    "(IP-Adapter FaceID PlusV2 + ControlNet Tile) "
                    "+ заливка нимба фоном (LaMa).")
    parser.add_argument("--image", required=True, help="Иллюстрация.")
    parser.add_argument("--client", required=True, help="Фото клиента.")
    parser.add_argument("--hair", default="",
                        help="Описание причёски (англ.), напр. 'long wavy "
                             "blonde hair'. Опционально, но помогает.")
    parser.add_argument("--person", default=None,
                        help="Описание клиента для промпта, напр. '40 year "
                             "old bearded man'. По умолчанию — авто по "
                             "возрасту/полу из InsightFace; --person '' "
                             "отключает подстановку.")
    parser.add_argument("--faceid", default=None, choices=["v1", "plusv2"],
                        help="Версия IP-Adapter FaceID (default: plusv2).")
    parser.add_argument("--base-model", default=None,
                        help="Другой SDXL-чекпоинт, напр. "
                             "RunDiffusion/Juggernaut-XL-v9 или "
                             "SG161222/RealVisXL_V5.0. По умолчанию "
                             "diffusers/sdxl-inpainting.")
    parser.add_argument("--no-bg-fill", action="store_true",
                        help="Отключить этап 3: заливку 'нимба' вокруг "
                             "головы фоном (LaMa).")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--ip-scale", type=float, default=None)
    parser.add_argument("--faceid-lora-scale", type=float, default=None,
                        help="Вес вспомогательной LoRA FaceID (0.5..1.0).")
    parser.add_argument("--control-scale", type=float, default=None,
                        help="Вес ControlNet Tile (0.3..0.5).")
    parser.add_argument("--paste-dilate", type=int, default=None,
                        help="Запас маски ВКЛЕЙКИ в px кропа gen_size "
                             "(default 6). Большие значения возвращают "
                             "'нимб' вокруг головы.")
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
    if args.paste_dilate is not None:
        overrides["paste_dilate"] = args.paste_dilate
    if args.gen_size is not None:
        overrides["gen_size"] = args.gen_size
    if args.faceid is not None:
        overrides["faceid_version"] = args.faceid
    if args.base_model is not None:
        overrides["base_model"] = args.base_model
    if args.no_bg_fill:
        overrides["bg_fill"] = False
    if args.lora is not None:
        overrides["lora_path"] = args.lora
    if args.lora_scale is not None:
        overrides["lora_scale"] = args.lora_scale
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

    # --- промпт: возраст/пол клиента ---
    if args.person is not None:
        person_desc = args.person.strip() or "person"
        extra_neg = ""
        if args.person.strip():
            print(f"Описание клиента (из --person): '{person_desc}'")
    else:
        person_desc, extra_neg = person_descriptor(cface)
        print(f"Авто-описание клиента: '{person_desc}' "
              f"(переопределить: --person).")
    prompt = cfg.prompt.replace("{person}", person_desc)
    if args.hair.strip():
        prompt = f"{prompt}, {args.hair.strip()}"
    negative = cfg.negative + (f", {extra_neg}" if extra_neg else "")
    cfg = dataclasses.replace(cfg, prompt=prompt, negative=negative)

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

    inpaint_region = col.inpaint_region.copy()

    # кроп считаем по региону головы: длинные волосы клиента попадают в кроп
    x1, y1, x2, y2 = mask_crop_box(inpaint_region, CROP_MARGIN, img_w, img_h)
    side = x2 - x1
    print(f"Кроп вокруг зоны: [{x1},{y1},{x2},{y2}] ({side}px) -> {cfg.gen_size}px")

    # ДВЕ маски (фикс внешнего "нимба"):
    #   gen_mask   — что перерисовывает модель (широкий запас на швы);
    #   paste_mask — что вклеиваем в оригинал (почти впритык к голове).
    # Кольцо перегенерированного фона между ними ОТБРАСЫВАЕТСЯ — вокруг
    # головы остаются оригинальные пиксели иллюстрации.
    gen_seam = max(1, round(cfg.seam_dilate * side / cfg.gen_size))
    paste_seam = max(0, round(cfg.paste_dilate * side / cfg.gen_size))
    gen_mask_full = dilate_px(inpaint_region, gen_seam)
    paste_mask_full = dilate_px(inpaint_region, paste_seam)

    crop_gen_mask = gen_mask_full[y1:y2, x1:x2]
    crop_paste_mask = paste_mask_full[y1:y2, x1:x2]
    crop_collage = col.collage_bgr[y1:y2, x1:x2]

    gs = cfg.gen_size
    crop_rgb = cv2.resize(cv2.cvtColor(crop_collage, cv2.COLOR_BGR2RGB),
                          (gs, gs), interpolation=cv2.INTER_LANCZOS4)
    mask_gs = cv2.resize(crop_gen_mask, (gs, gs), interpolation=cv2.INTER_NEAREST)
    if cfg.mask_blur > 0:
        k = cfg.mask_blur * 2 + 1
        mask_gs = cv2.GaussianBlur(mask_gs, (k, k), 0)

    pil_image = Image.fromarray(crop_rgb)     # база инпейнта — коллаж
    pil_control = Image.fromarray(crop_rgb)   # control image (Tile) — тоже коллаж
    pil_mask = Image.fromarray(mask_gs)

    # ====================== ЭТАП 2: ИНПЕЙНТ + CONTROLNET =====================
    pipe = build_pipeline(device, cfg)
    pipe.set_ip_adapter_scale(cfg.ip_scale)
    if cfg.faceid_version == "plusv2":
        setup_faceid_plusv2(pipe, client_bgr, cface, device, dtype)
    id_embeds = faceid_embeds_for_pipe(ref_embed, device, dtype)

    # =========== ЭТАП 3 (подготовка): заливка нимба фоном (LaMa) ===========
    # Кольцо между новой (меньшей) головой и старым контуром персонажа лежит
    # ВНУТРИ paste-маски — там диффузия рисует псевдо-фон ("нимб").
    # BackgroundFiller: face-parsing защищает лицо/волосы/ШЕЮ/одежду нового
    # персонажа, остаток региона заливается LaMa настоящим фоном.
    bg_filler = None
    if cfg.bg_fill:
        print("Этап 3: загрузка face-parsing + LaMa для заливки нимба...")
        bg_filler = BackgroundFiller(device)
    else:
        print("Этап 3 (заливка нимба) отключён (--no-bg-fill).")

    out_dir = OUTPUTS_DIR / f"{stem}__{client_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "collage.png"), col.collage_bgr)
    cv2.imwrite(str(out_dir / "inpaint_mask.png"), gen_mask_full)
    cv2.imwrite(str(out_dir / "paste_mask.png"), paste_mask_full)

    # мягкая альфа для финальной вклейки — по УЗКОЙ paste-маске, размытие
    # пересчитано из px gen_size в px кропа (раньше не масштабировалось)
    blur_px = max(1, round(cfg.mask_blur * side / cfg.gen_size))
    kb = 2 * blur_px + 1
    alpha_full = cv2.GaussianBlur(crop_paste_mask, (kb, kb), 0)
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

        # кроп gen_size -> обратно в размер кропа -> мягкая вклейка по
        # УЗКОЙ маске на ОРИГИНАЛ иллюстрации (фон вокруг головы не трогаем)
        gen_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.resize(gen_bgr, (side, side), interpolation=cv2.INTER_LANCZOS4)
        composed = illus_bgr.copy()
        region = composed[y1:y2, x1:x2].astype(np.float32)
        blended = alpha_full * gen_bgr.astype(np.float32) + (1 - alpha_full) * region
        composed[y1:y2, x1:x2] = blended.astype(np.uint8)

        # ==================== ЭТАП 3: ЗАЛИВКА НИМБА =========================
        bg_status = "disabled"
        halo_mask = None
        if bg_filler is not None:
            composed, halo_mask, bg_status = bg_filler(composed, paste_mask_full)
            print(f"  заливка нимба: {bg_status}")

        sim = face_similarity(face_app, composed[y1:y2, x1:x2], ref_embed)
        print(f"  similarity={sim:.2f}" + ("  (лицо не найдено!)" if sim < 0 else ""))

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

    # quality gate: если лучший кандидат без детектируемого лица — это симптом
    # "лицо слилось с фоном". Явно сигналим и подсказываем, что крутить.
    if best["similarity"] < 0.15:
        print("\n[!] ВНИМАНИЕ: на лучшем кандидате лицо почти/совсем не "
              "детектируется (similarity низкий). Вероятно лицо слилось с "
              "фоном/потеряло структуру. Попробуйте усилить якорь лица:\n"
              "    --control-scale 0.5  (Tile сильнее держит структуру лица)\n"
              "    --ip-scale 1.0       (сильнее переносит личность клиента)\n"
              "    и проверьте wipe_method в логе: 'geometric' — самый грубый, "
              "возможно, маска головы (illustrations_mask) неточная.")

    grid_path = out_dir / "grid.jpg"
    make_grid(candidates, (x1, y1, x2, y2), grid_path)

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "variant": "v3.2_collage_controlnet_plusv2_bgfill",
            "image": str(image_path), "client": str(client_path),
            "mask": str(mask_path), "crop_box": [x1, y1, x2, y2],
            "faceid_version": cfg.faceid_version,
            "base_model": cfg.base_model or SDXL_INPAINT_ID,
            "controlnet": CONTROLNET_TILE_ID,
            "person": person_desc,
            "align_method": col.align_method, "hair_pasted": col.hair_pasted,
            "face_preserved": col.face_preserved, "wipe_method": col.wipe_method,
            "paste_face": bool(args.paste_face), "hair": args.hair,
            "bg_fill": cfg.bg_fill,
            "prompt": cfg.prompt, "negative": cfg.negative,
            "ip_scale": cfg.ip_scale, "control_scale": cfg.control_scale,
            "strength": cfg.strength, "steps": cfg.steps,
            "guidance": cfg.guidance, "gen_size": cfg.gen_size,
            "seam_dilate": cfg.seam_dilate, "paste_dilate": cfg.paste_dilate,
            "mask_blur": cfg.mask_blur,
            "lora": cfg.lora_path, "lora_scale": cfg.lora_scale,
            "candidates": [{"seed": c["seed"], "similarity": c["similarity"],
                            "path": c["path"], "bg_fill": c["bg_status"]}
                           for c in candidates],
            "best_seed": best["seed"], "best_similarity": best["similarity"],
            "best_bg_fill": best["bg_status"],
            "best_path": str(best_path),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nЛучший seed: {best['seed']} (similarity={best['similarity']:.2f}, "
          f"нимб: {best['bg_status']})")
    print(f"Итог:   {best_path}")
    print(f"Коллаж: {out_dir / 'collage.png'}")
    print(f"Сетка:  {grid_path}")
    print(f"JSON:   {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()