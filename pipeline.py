"""
Оркестратор пайплайна: Иллюстрация + Фото клиента -> Лучший результат.

Запускает по шагам:
  1) illustrations_detect.py   — детекция лица на иллюстрации (RetinaFace)
  2) illustrations_mask.py     — маска лицо+волосы (SAM2)
  3) output_inpainting_v3.py   — коллаж волос + SDXL inpaint (ControlNet + FaceID)

Маска на иллюстрации по дизайну проекта СТАТИЧНА: считается один раз на сцену
и переиспользуется для всех клиентов. Поэтому шаги 1-2 по умолчанию
ПРОПУСКАЮТСЯ, если маска для этой иллюстрации уже посчитана.
Пересчитать принудительно: --force-mask.

Запуск:
  python pipeline.py --image data/illustrations/spread_01.png \
                     --client data/clients/ivan.jpg \
                     --hair "long wavy blonde hair" \
                     --output final_result.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(cmd: list[str], step_name: str) -> None:
    """Запуск шага тем же интерпретатором, из корня проекта, без shell.

    Список аргументов (а не строка + shell=True) корректно обрабатывает
    пробелы в путях и в описании причёски — ничего экранировать не нужно.
    """
    print(f"\n[{step_name}] Запуск: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        print(f"\n[ОШИБКА] {step_name} завершился с ошибкой.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Пайплайн: Иллюстрация + Клиент -> Лучший результат."
    )
    parser.add_argument("--image", required=True, help="Путь к иллюстрации.")
    parser.add_argument("--client", required=True, help="Путь к фото клиента.")
    parser.add_argument("--hair", default="",
                        help="Описание причёски (англ., опционально).")
    parser.add_argument("--output", default="final_result.png",
                        help="Куда сохранить итоговый файл.")
    parser.add_argument("--force-mask", action="store_true",
                        help="Пересчитать детекцию+маску, даже если они уже есть.")
    parser.add_argument("--keep-outputs", action="store_true",
                        help="Не удалять папку data/outputs/<scene>__<client> "
                             "(там grid.jpg, кандидаты, result.json).")
    parser.add_argument("--clean-mask", action="store_true",
                        help="В конце удалить также детекцию+маску. По умолчанию "
                             "они СОХРАНЯЮТСЯ для переиспользования (маска "
                             "статична для всех клиентов этой сцены).")
    args = parser.parse_args()

    py = sys.executable  # тот же venv/интерпретатор, что запустил pipeline

    image_path = Path(args.image).resolve()
    client_path = Path(args.client).resolve()
    final_output = Path(args.output).resolve()

    for p, what in [(image_path, "иллюстрация"), (client_path, "фото клиента")]:
        if not p.exists():
            print(f"[ОШИБКА] Не найден файл ({what}): {p}")
            sys.exit(1)

    stem = image_path.stem
    client_stem = client_path.stem

    det_json = PROJECT_ROOT / "data" / "detections" / f"{stem}.json"
    det_vis = PROJECT_ROOT / "data" / "detections" / f"{stem}_vis.jpg"
    mask_png = PROJECT_ROOT / "data" / "masks" / f"{stem}_mask.png"
    mask_vis = PROJECT_ROOT / "data" / "masks" / f"{stem}_vis.jpg"
    mask_json = PROJECT_ROOT / "data" / "masks" / f"{stem}.json"
    output_dir = PROJECT_ROOT / "data" / "outputs" / f"{stem}__{client_stem}"

    # ── Шаги 1-2: детекция + маска (статичны, считаются один раз на сцену) ──
    if args.force_mask or not mask_png.exists():
        run_step([py, "illustrations_detect.py", "--image", str(image_path)],
                 "ЭТАП 1: Детекция")
        run_step([py, "illustrations_mask.py", "--image", str(image_path)],
                 "ЭТАП 2: Маскирование")
    else:
        print(f"\n[ПРОПУСК] Маска уже посчитана: {mask_png}\n"
              f"          (пересчитать: --force-mask)")

    # ── Шаг 3: коллаж волос клиента + SDXL inpaint (ControlNet + FaceID) ──
    cmd = [py, "output_inpainting_v3.py",
           "--image", str(image_path), "--client", str(client_path)]
    if args.hair:
        cmd += ["--hair", args.hair]  # список -> пробелы внутри строки безопасны
    run_step(cmd, "ЭТАП 3: Генерация")

    # ── Сборка результата ──
    print("\n[СБОРКА] Извлечение лучшего результата...")
    best_image = output_dir / "best.png"
    if best_image.exists():
        final_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_image, final_output)
        print(f"[УСПЕХ] Финальное изображение сохранено: {final_output}")
    else:
        print("[ОШИБКА] best.png не найден — сбой на этапе генерации.")
        sys.exit(1)

    # ── Очистка ──
    # Временная, специфичная для клиента папка — удаляется по умолчанию.
    if not args.keep_outputs and output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[ОЧИСТКА] Удалена временная папка: {output_dir}")

    # Детекцию и маску по умолчанию НЕ трогаем — они переиспользуются
    # следующими клиентами этой же сцены (иначе SAM2 гоняется впустую).
    if args.clean_mask:
        for f in [det_json, det_vis, mask_png, mask_vis, mask_json]:
            if f.exists():
                f.unlink()
        print("[ОЧИСТКА] Детекция и маска удалены (--clean-mask).")
    else:
        print("[ИНФО] Детекция и маска сохранены для переиспользования "
              "(удалить: --clean-mask).")


if __name__ == "__main__":
    main()