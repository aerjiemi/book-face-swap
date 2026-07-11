import argparse
import subprocess
import shutil
import sys
from pathlib import Path


def run_step(command: str, step_name: str):
    print(f"\n[{step_name}] Запуск...")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ОШИБКА] {step_name} завершился с ошибкой.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Clean Pipeline: Иллюстрация + Клиент -> Лучший результат")
    parser.add_argument("--image", required=True, help="Путь к иллюстрации")
    parser.add_argument("--client", required=True, help="Путь к фото клиента")
    parser.add_argument("--hair", default="", help="Описание прически (опционально)")
    parser.add_argument("--output", default="final_result.png", help="Куда сохранить итоговый файл")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    client_path = Path(args.client).resolve()
    final_output = Path(args.output).resolve()
    stem = image_path.stem
    client_stem = client_path.stem

    # Шаг 1: Детекция лица (RetinaFace)
    run_step(f"python illustrations_detect.py --image {image_path}", "ЭТАП 1: Детекция")

    # Шаг 2: Создание маски (SAM2)
    run_step(f"python illustrations_mask.py --image {image_path}", "ЭТАП 2: Маскирование")

    # Шаг 3: Коллаж и Инпейнт (SDXL + ControlNet)
    hair_arg = f'--hair "{args.hair}"' if args.hair else ""
    run_step(
        f"python output_inpainting_v3.py --image {image_path} --client {client_path} {hair_arg}",
        "ЭТАП 3: Генерация"
    )

    # Шаг 4: Извлечение лучшего результата и очистка мусора
    print("\n[ОЧИСТКА] Сборка лучшего результата и удаление временных файлов...")

    # Пути к сгенерированному мусору
    proj_root = Path(__file__).resolve().parent
    det_json = proj_root / "data" / "detections" / f"{stem}.json"
    det_vis = proj_root / "data" / "detections" / f"{stem}_vis.jpg"
    mask_png = proj_root / "data" / "masks" / f"{stem}_mask.png"
    mask_vis = proj_root / "data" / "masks" / f"{stem}_vis.jpg"
    mask_json = proj_root / "data" / "masks" / f"{stem}.json"
    output_dir = proj_root / "data" / "outputs" / f"{stem}__{client_stem}"

    best_image = output_dir / "best.png"

    if best_image.exists():
        # Копируем лучший результат в указанное место
        shutil.copy2(best_image, final_output)
        print(f"\n[УСПЕХ] Финальное изображение сохранено: {final_output}")
    else:
        print("\n[ОШИБКА] Файл best.png не найден. Что-то пошло не так на этапе генерации.")

    # Удаляем временные файлы
    for f in [det_json, det_vis, mask_png, mask_vis, mask_json]:
        if f.exists():
            f.unlink()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    print("[ОЧИСТКА] Временные файлы удалены.")


if __name__ == "__main__":
    main()