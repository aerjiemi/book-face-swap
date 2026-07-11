"""
В этом файле находится этап 1 пайплайна.

Как работает:
1) Берет иллюстрацию книги из data/illustrations
2) с помощью RetinaFace находит лицо -> bbox [x1, y1, x2, y2], landmarks (5 points), score
3) Результат сохраняется в виде:
JSON (data/detections/<name>.json),
Картинка с bbox для визуальной проверки (data/detections/<name>_vis.jpg)


Запуск:
  python illustrations_detect.py                  # выбрать из списка вручную
  python illustrations_detect.py --image path.png # конкретный файл

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from retinaface import RetinaFace


PROJECT_ROOT = Path(__file__).resolve().parent
ILLUSTRATIONS_DIR = PROJECT_ROOT / "data" / "illustrations"
DETECTIONS_DIR = PROJECT_ROOT / "data" / "detections"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_illustrations() -> list[Path]:
    """Возвращает отсортированный по имени список всех изображений в data/illustrations."""
    return sorted(
        p for p in ILLUSTRATIONS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def choose_illustration() -> Path:
    """Ручной выбор иллюстрации из списка по номеру."""
    files = list_illustrations()
    if not files:
        raise FileNotFoundError(f"В {ILLUSTRATIONS_DIR} нет картинок.")
    print("Доступные иллюстрации:")
    for i, p in enumerate(files):
        print(f"  [{i}] {p.name}")
    while True:
        raw = input("Введи номер иллюстрации: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(files):
            return files[int(raw)]
        print("Некорректный номер, попробуй ещё раз.")


def detect_faces(image_path: Path, threshold: float = 0.5) -> list[dict]:
    """
    Запускает RetinaFace и возвращает список детекций, отсортированный
    по убыванию confidence.

    Каждая детекция (JSON):
        {
            "score": float,
            "bbox": [x1, y1, x2, y2],       # int, пиксели
            "landmarks": {                   # 5 точек, [x, y]
                "right_eye": [...], "left_eye": [...], "nose": [...],
                "mouth_right": [...], "mouth_left": [...]
            }
        }

    При необходимости можно менять threshold (default=0.5)

    """

    result = RetinaFace.detect_faces(str(image_path), threshold=threshold)

    # При отсутствии лиц
    if not isinstance(result, dict) or not result:
        return []

    # Формируем список детекций
    detections = []
    for face in result.values():
        x1, y1, x2, y2 = face["facial_area"]
        detections.append({
            "score": float(face["score"]),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "landmarks": {
                name: [float(pt[0]), float(pt[1])]
                for name, pt in face["landmarks"].items()
            },
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def draw_visualization(image_path: Path, detections: list[dict], out_path: Path) -> None:
    """ Рисует визуализацию с bbox и landmarks для проверки """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f"{det['score']:.2f}", (x1, max(12, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        for px, py in det["landmarks"].values():
            cv2.circle(img, (int(px), int(py)), 2, (0, 0, 255), -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Детекция лица на иллюстрации (RetinaFace)."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Путь к конкретной иллюстрации. Если не задан — выбор из списка.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Порог уверенности детектора.",
    )
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else choose_illustration()
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    print(f"\nВыбрано изображение: {image_path.name} ")
    detections = detect_faces(image_path, threshold=args.threshold)

    if not detections:
        print("Лицо не найдено.") # в таком случае можно просить пользователя прислать другое фото
        return

    stem = image_path.stem

    json_path = DETECTIONS_DIR / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(image_path),
                "threshold": args.threshold,
                "detections": detections,
            },
            f, ensure_ascii=False, indent=2,
        )

    vis_path = DETECTIONS_DIR / f"{stem}_vis.jpg"
    draw_visualization(image_path, detections, vis_path)



if __name__ == "__main__":
    main()