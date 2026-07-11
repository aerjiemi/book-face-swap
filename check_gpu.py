"""
check_gpu.py

Диагностика окружения под RTX 5050 Laptop (Blackwell, sm_120).

Проверяет по цепочке:
  1. Видит ли система карту вообще (nvidia-smi, драйвер).
  2. Какая сборка torch стоит (CPU-сборка? под какую CUDA собрана?).
  3. Видит ли torch карту и умеет ли её архитектуру (sm_120).
  4. Реальный тестовый прогон на GPU (matmul в fp16 и bf16).
  5. Каким провайдером работает onnxruntime (для InsightFace).

На каждом падении печатает, ЧТО именно ставить/чинить.

Запуск:  python check_gpu.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys


OK = "  [OK] "
FAIL = "  [FAIL] "
WARN = "  [!] "

# Blackwell (RTX 50xx) = compute capability 12.0
BLACKWELL_CC = (12, 0)

FIX_INSTALL = (
    "\n  Как чинить (RTX 5050 = Blackwell, ей нужен torch с поддержкой sm_120):\n"
    "    pip uninstall -y torch torchvision torchaudio\n"
    "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
    "  (нужен PyTorch 2.7+ на CUDA 12.8+; более новые cu-индексы тоже подходят)\n"
    "  И проверь, что драйвер NVIDIA свежий (R570+): смотри поле Driver Version в nvidia-smi."
)


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def check_driver() -> bool:
    step("1. Драйвер / nvidia-smi")
    if shutil.which("nvidia-smi") is None:
        print(FAIL + "nvidia-smi не найден. Драйвер NVIDIA не установлен либо не в PATH.")
        print("  Поставь/обнови драйвер NVIDIA (для RTX 50xx нужен R570 или новее).")
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            print(FAIL + f"nvidia-smi вернул ошибку:\n{out.stderr.strip()}")
            return False
        for line in out.stdout.strip().splitlines():
            print(OK + line.strip())
        return True
    except Exception as e:
        print(FAIL + f"Не удалось выполнить nvidia-smi: {e}")
        return False


def check_torch_build():
    step("2. Сборка PyTorch")
    try:
        import torch
    except ImportError:
        print(FAIL + "torch не установлен." + FIX_INSTALL)
        return None
    print(OK + f"torch {torch.__version__}")
    cuda_built = torch.version.cuda
    if cuda_built is None:
        print(FAIL + "Это CPU-сборка torch (torch.version.cuda = None). "
                     "Именно поэтому генерация идёт на процессоре." + FIX_INSTALL)
        return torch
    print(OK + f"Сборка под CUDA {cuda_built}")
    try:
        major = int(str(cuda_built).split(".")[0])
        minor = int(str(cuda_built).split(".")[1])
        if (major, minor) < (12, 8):
            print(WARN + f"CUDA {cuda_built} старее 12.8 — ядра под sm_120 (RTX 50xx) "
                         "в этой сборке отсутствуют." + FIX_INSTALL)
    except (ValueError, IndexError):
        pass
    return torch


def check_torch_sees_gpu(torch) -> bool:
    step("3. torch.cuda")
    if not torch.cuda.is_available():
        print(FAIL + "torch.cuda.is_available() == False — torch карту не видит."
              + FIX_INSTALL)
        return False
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cc = torch.cuda.get_device_capability(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(OK + f"cuda:{i} {name} | compute capability {cc[0]}.{cc[1]} | "
                   f"{vram:.1f} GB VRAM")
        if cc >= BLACKWELL_CC:
            archs = torch.cuda.get_arch_list()
            if not any("120" in a for a in archs):
                print(FAIL + f"Карта Blackwell (sm_{cc[0]}{cc[1]}), но в сборке torch "
                             f"нет ядер под неё. Скомпилированные архитектуры: {archs}."
                      + FIX_INSTALL)
                return False
            print(OK + f"Ядра под sm_120 в сборке есть: {[a for a in archs if '12' in a]}")
    return True


def check_gpu_compute(torch) -> bool:
    step("4. Тестовый прогон на GPU")
    try:
        for dtype, name in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
            a = torch.randn(1024, 1024, device="cuda", dtype=dtype)
            b = torch.randn(1024, 1024, device="cuda", dtype=dtype)
            c = (a @ b).sum().item()
            print(OK + f"matmul 1024x1024 в {name}: ок (sum={c:.1f})")
        free, total = torch.cuda.mem_get_info()
        print(OK + f"VRAM свободно: {free / 1024**3:.1f} / {total / 1024**3:.1f} GB")
        return True
    except RuntimeError as e:
        msg = str(e)
        print(FAIL + f"Ошибка при вычислении на GPU: {msg[:300]}")
        if "no kernel image" in msg.lower():
            print("  Это классический признак: сборка torch не содержит ядер "
                  "под архитектуру карты (sm_120)." + FIX_INSTALL)
        return False


def check_onnxruntime() -> None:
    step("5. onnxruntime (нужен InsightFace)")
    try:
        import onnxruntime as ort
    except ImportError:
        print(WARN + "onnxruntime не установлен. Для InsightFace: "
                     "pip install onnxruntime-gpu (или onnxruntime для CPU).")
        return
    providers = ort.get_available_providers()
    print(OK + f"onnxruntime {ort.__version__}, провайдеры: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print(WARN + "CUDA-провайдера нет — InsightFace будет считать на CPU. "
                     "Для эмбеддингов это НЕ критично (они быстрые и на CPU); "
                     "тяжёлая часть — SDXL — идёт через torch.")


def main() -> None:
    print("Диагностика GPU для пайплайна (ожидается RTX 5050 Laptop / Blackwell).")
    print(f"Python: {sys.version.split()[0]}")

    driver_ok = check_driver()
    torch_mod = check_torch_build()
    if torch_mod is None:
        sys.exit(1)

    sees = check_torch_sees_gpu(torch_mod)
    compute_ok = check_gpu_compute(torch_mod) if sees else False
    check_onnxruntime()

    step("Итог")
    if driver_ok and sees and compute_ok:
        print(OK + "GPU полностью рабочий. output_inpainting.py подхватит его сам.")
    else:
        print(FAIL + "GPU не готов — смотри [FAIL] выше и инструкции по установке.")
        sys.exit(1)


if __name__ == "__main__":
    main()