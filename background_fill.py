"""
background_fill.py  — заливка "нимба" вокруг новой головы одним
SDXL-inpaint проходом. Геометрическая детекция нимба
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

MIN_HALO_FRAC = 0.005   # кольцо меньше 0.5% региона — заливать нечего
MIN_HEAD_FRAC = 0.25    # парсинг должен покрыть >= 25% старой головы,
                        # иначе считаем его проваленным и не рискуем лицом
SKIN_COLOR_DELTA = 30   # страховка подбородка: |пиксель - медианный цвет
                        # кожи| < delta (по максимальному каналу) -> защита
SKIN_GUARD_BAND_FRAC = 0.12  # ширина полосы skin-страховки вокруг
                             # распарсенной головы (доля от размера региона)

# классы face-parsing (CelebAMask-HQ), которые НИКОГДА не закрашиваются
PROTECTED_CLASSES = {
    "skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "eye_g",
    "u_lip", "l_lip", "mouth", "l_ear", "r_ear", "ear_r", "hair",
    "neck", "neck_l", "cloth", "hat",
}


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Заливает полностью замкнутые дыры внутри маски (флудфилл от угла)."""
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
    return cv2.bitwise_or(mask_u8, cv2.bitwise_not(flood))


class BackgroundFiller:
    """Заливка нимба одним SDXL-inpaint проходом по кольцу вокруг головы."""

    def __init__(self, device: str, pipe,
                 gen_size: int = 1024,
                 strength: float = 1.0,
                 steps: int = 40,
                 guidance: float = 6.0,
                 control_scale: float = 0.0,
                 protect_grow_frac: float = 0.04,
                 halo_grow_px: int = 12,
                 restore_ip_scale: float = 1.0,
                 parser=None):
        self.device = device
        self.pipe = pipe
        self.gen_size = gen_size
        self.strength = strength
        self.steps = steps
        self.guidance = guidance
        self.control_scale = control_scale
        self.protect_grow_frac = protect_grow_frac
        self.halo_grow_px = halo_grow_px
        self._restore_ip_scale = restore_ip_scale
        self._ref_embed_shape = None
        self.parser = parser

        self.prompt = (
            "seamless continuation of the surrounding background, canvas texture, "
            "same art style, same colors, visible thick brush strokes as the "
            "surrounding oil painting, coherent background scenery, no person, no face")
        self.negative = (
            "face, head, portrait, person, eyes, hair, skin, neck, halo, glow, ring, "
            "vignette, bright outline, seam, border, smooth flat blob, grey mask, "
            "blurry, low quality, artifact, photo, photorealistic, plastic")

    def set_ref_embed_shape(self, shape) -> None:
        self._ref_embed_shape = tuple(shape)

    # ------------------------- защита новой головы ---------------------------

    def _parsed_protection(self, image_bgr: np.ndarray, region255: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Маски (защищаемая голова, кожа) новой головы на готовом кадре.

        Парсинг делается на зум-кропе вокруг региона: SegFormer на
        стилизованном лице в полном кадре ненадёжен (это и было причиной
        пропадающих подбородков). Результат ограничивается окрестностью
        региона, чтобы не цеплять других персонажей сцены.
        """
        from hair_collage import _parse_labels, _labels_to_mask

        h, w = image_bgr.shape[:2]
        ys, xs = np.where(region255 > 0)
        pad = int(0.30 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 8
        cx1, cy1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        cx2, cy2 = min(w, int(xs.max()) + pad), min(h, int(ys.max()) + pad)

        labels_map = _parse_labels(image_bgr[cy1:cy2, cx1:cx2], self.parser)
        id2label = self.parser[2]
        parsed_c = _labels_to_mask(labels_map, id2label, PROTECTED_CLASSES)
        skin_c = _labels_to_mask(labels_map, id2label, {"skin"})

        parsed = np.zeros((h, w), np.uint8)
        skin = np.zeros((h, w), np.uint8)
        parsed[cy1:cy2, cx1:cx2] = parsed_c
        skin[cy1:cy2, cx1:cx2] = skin_c

        guard = cv2.dilate(region255, _ellipse(31))
        parsed = cv2.bitwise_and(parsed, guard)
        skin = cv2.bitwise_and(skin, guard)

        # сшиваем разрывы и заливаем дыры (глаза/блики) — внутри лица
        # "фона" быть не может по определению
        parsed = cv2.morphologyEx(parsed, cv2.MORPH_CLOSE, _ellipse(9))
        parsed = _fill_holes(parsed)
        return parsed, skin

    # --------------------------- SDXL-inpaint --------------------------------

    def _run_inpaint(self, crop_bgr: np.ndarray, halo_crop: np.ndarray) -> np.ndarray:
        """SDXL-inpaint по маске halo_crop (identity выключен на время прохода)."""
        from PIL import Image

        gs = self.gen_size
        h0, w0 = crop_bgr.shape[:2]

        crop_rgb = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                              (gs, gs), interpolation=cv2.INTER_LANCZOS4)
        mask_gs = cv2.resize(halo_crop, (gs, gs), interpolation=cv2.INTER_NEAREST)
        mask_gs = cv2.GaussianBlur(mask_gs, (11, 11), 0)

        pil_image = Image.fromarray(crop_rgb)
        pil_control = Image.fromarray(crop_rgb)
        pil_mask = Image.fromarray(mask_gs)

        shape = self._ref_embed_shape or (2, 1, 512)
        zero_embeds = torch.zeros(
            shape, device=torch.device(self.device),
            dtype=(torch.float16 if self.device == "cuda" else torch.float32))

        try:
            self.pipe.set_ip_adapter_scale(0.0)
            gen = torch.Generator(device="cpu").manual_seed(42)
            result = self.pipe(
                prompt=self.prompt,
                negative_prompt=self.negative,
                image=pil_image,
                mask_image=pil_mask,
                control_image=pil_control,
                controlnet_conditioning_scale=0.0,
                ip_adapter_image_embeds=[zero_embeds],
                strength=self.strength,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                generator=gen,
                height=gs, width=gs,
            ).images[0]
        finally:
            self.pipe.set_ip_adapter_scale(self._restore_ip_scale)

        out = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LANCZOS4)

    # ------------------------------ основной вызов ---------------------------

    def __call__(self, image_bgr: np.ndarray, region_mask: np.ndarray,
                 head_region: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, str]:
        """image_bgr    — кадр после основной генерации;
        region_mask  — paste-регион (старая голова + небольшой запас);
        head_region  — маска старой головы (для оценки провала парсинга).
        """
        if self.parser is None:
            from hair_collage import build_face_parser
            self.parser = build_face_parser(self.device)

        h, w = image_bgr.shape[:2]
        region255 = ((region_mask > 0).astype(np.uint8)) * 255
        region_area = max(int((region255 > 0).sum()), 1)
        head_area = (max(int((head_region > 0).sum()), 1)
                     if head_region is not None else region_area)
        empty = np.zeros((h, w), np.uint8)

        # --- 1. защита: face-parsing НОВОЙ головы (зум-кроп) ---
        parsed, skin = self._parsed_protection(image_bgr, region255)

        # fail-safe: парсинг провалился -> не рисковать лицом, нимб оставить
        inside = cv2.bitwise_and(parsed, region255)
        if int((inside > 0).sum()) < MIN_HEAD_FRAC * head_area:
            return image_bgr, empty, "skipped_parse_failed"

        # --- 2. расширение защиты (protect_grow_frac от размера региона) ---
        ys, xs = np.where(region255 > 0)
        size = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()), 1)
        protect_px = max(3, int(round(self.protect_grow_frac * size)))
        protection = cv2.dilate(parsed, _ellipse(2 * protect_px + 1))

        # --- 3. доп. цветовой фильтр-СТРАХОВКА
        if skin.any():
            med = np.median(image_bgr[skin > 0].reshape(-1, 3), axis=0)
            diff = np.abs(image_bgr.astype(np.int16)
                          - med.astype(np.int16)).max(axis=2)
            skin_like = ((diff < SKIN_COLOR_DELTA).astype(np.uint8)) * 255
            band_px = max(3 * protect_px,
                          int(round(SKIN_GUARD_BAND_FRAC * size)))
            band = cv2.dilate(parsed, _ellipse(2 * band_px + 1))
            skin_guard = cv2.bitwise_and(skin_like, band)
            protection = cv2.bitwise_or(
                protection, cv2.bitwise_and(skin_guard, region255))
        protection = _fill_holes(protection)

        # --- 4. нимб = регион МИНУС защита (геометрически, без цвета) ---
        halo = cv2.subtract(region255, protection)
        halo = cv2.morphologyEx(halo, cv2.MORPH_OPEN, _ellipse(3))

        # настоящее кольцо всегда касается внешней границы региона;
        # изолированные "дыры" парсера внутри лица — не нимб, не трогаем
        border = cv2.subtract(region255, cv2.erode(region255, _ellipse(7)))
        num, lab, _stats, _ = cv2.connectedComponentsWithStats(
            (halo > 0).astype(np.uint8), connectivity=8)
        if num > 1:
            keep = [i for i in range(1, num)
                    if np.any((lab == i) & (border > 0))]
            halo = (np.isin(lab, keep).astype(np.uint8)) * 255

        if int((halo > 0).sum()) < MIN_HALO_FRAC * region_area:
            return image_bgr, halo, "skipped_no_halo"

        # --- 5. маска генерации: расширяем кольцо, но защиту не трогаем ---
        halo_gen = cv2.dilate(halo, _ellipse(2 * self.halo_grow_px + 1))
        halo_gen = cv2.subtract(halo_gen, protection)

        ys, xs = np.where(halo_gen > 0)
        if len(ys) == 0:
            return image_bgr, halo, "skipped_empty_mask"

        pad = int(0.35 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 32
        cx1 = max(0, int(xs.min()) - pad); cy1 = max(0, int(ys.min()) - pad)
        cx2 = min(w, int(xs.max()) + pad); cy2 = min(h, int(ys.max()) + pad)

        filled_crop = self._run_inpaint(image_bgr[cy1:cy2, cx1:cx2],
                                        halo_gen[cy1:cy2, cx1:cx2])

        # --- 6. вклейка: альфа жёстко обнулена на каждом пикселе защиты ---
        alpha = cv2.GaussianBlur(halo_gen[cy1:cy2, cx1:cx2],
                                 (7, 7), 0).astype(np.float32) / 255.0
        alpha[protection[cy1:cy2, cx1:cx2] > 0] = 0.0
        alpha = alpha[..., None]

        out = image_bgr.copy()
        base = out[cy1:cy2, cx1:cx2].astype(np.float32)
        out[cy1:cy2, cx1:cx2] = (
            alpha * filled_crop.astype(np.float32) + (1.0 - alpha) * base
        ).astype(np.uint8)

        return out, halo_gen, "inpaint"