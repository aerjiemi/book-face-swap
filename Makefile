.PHONY: build shell run

# Сборка образа
build:
	docker build -t face-swap-env .

# Запуск интерактивного шелла (для отладки)
shell:
	docker run --rm -it \
		--gpus all \
		-v $(PWD):/app \
		-v $(PWD)/_models_cache/huggingface:/root/.cache/huggingface \
		-v $(PWD)/_models_cache/insightface:/root/.insightface \
		face-swap-env bash

# Запуск пайплайна (пример: make run IMAGE=data/illustrations/spread_01.png CLIENT=data/clients/ivan.jpg)
run:
	docker run --rm \
		--gpus all \
		-v $(PWD):/app \
		-v $(PWD)/_models_cache/huggingface:/root/.cache/huggingface \
		-v $(PWD)/_models_cache/insightface:/root/.insightface \
		face-swap-env python pipeline.py --image $(IMAGE) --client $(CLIENT) --output final_result.png