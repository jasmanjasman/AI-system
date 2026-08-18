# Makefile — single human interface, like LoopLM
# Run `make` (or `make help`) to see everything available.

.DEFAULT_GOAL := help
.PHONY: help install config check-llama llama run health chat docs up down logs ps rebuild

# Single source of truth: .env. Missing file is fine — the defaults below apply.
# .env wins over the shell environment, so for a one-off override use make's own
# syntax:  make health APP_PORT=9999   (NOT  APP_PORT=9999 make health)
-include .env
export

# ?= only fills in what .env did not already define.
LLAMA_BASE_URL   ?= http://localhost:8085/v1
LLAMA_MODEL      ?= qwen3
APP_HOST         ?= 127.0.0.1
APP_PORT         ?= 8001

# Used only by the `llama` target, to start the model server itself.
LLAMA_DIR        ?= $(HOME)/llama-cpp-model
LLAMA_MODEL_FILE ?= ./models/qwen3-1.7b.gguf
LLAMA_PORT       ?= 8085
# 0.0.0.0 is required for the Docker path to reach the host; it also exposes the
# model server to your local network. Set to 127.0.0.1 if you only use `make run`.
LLAMA_BIND       ?= 0.0.0.0

APP_URL := http://$(APP_HOST):$(APP_PORT)

DOWN_MSG = "App is not running at $(APP_URL) — start it with 'make run' (local) or 'make up' (docker)."
LLAMA_DOWN_MSG = "Model server is not running at $(LLAMA_BASE_URL) — start it with 'make llama'."

help: ## Show this help
	@echo "toolbox — available commands:"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Typical flow:  make llama   (terminal 1)"
	@echo "               make run     (terminal 2)"
	@echo "               make chat    (terminal 3)"

install: ## Install Python dependencies
	pip install -r requirements.txt

config: ## Show effective settings (run this first when something looks wrong)
	@echo "LLAMA_BASE_URL = $(LLAMA_BASE_URL)"
	@echo "LLAMA_MODEL    = $(LLAMA_MODEL)"
	@echo "APP_URL        = $(APP_URL)"

llama: ## Start the llama.cpp model server (occupies this terminal)
	cd $(LLAMA_DIR) && llama-server \
		--model $(LLAMA_MODEL_FILE) \
		--alias $(LLAMA_MODEL) \
		--port $(LLAMA_PORT) \
		--host $(LLAMA_BIND) \
		--n-gpu-layers -1 \
		--ctx-size 8192 \
		--batch-size 1024 \
		--threads 4 \
		--jinja

check-llama: ## Check the model server is up and speaking the OpenAI API
	@curl -s --max-time 5 $(LLAMA_BASE_URL)/models && echo \
		|| echo $(LLAMA_DOWN_MSG)

run: ## Start the app locally (occupies this terminal)
	uvicorn app.main:app --reload --host $(APP_HOST) --port $(APP_PORT)

health: ## Check the app is responding
	@curl -s --max-time 5 $(APP_URL)/health && echo \
		|| echo $(DOWN_MSG)

chat: ## Send a test prompt through the app (exercises the tool loop)
	@curl -s --max-time 120 -X POST $(APP_URL)/chat \
		-H "Content-Type: application/json" \
		-d '{"prompt": "What is 23 * 47?"}' && echo \
		|| echo $(DOWN_MSG)

docs: ## Print the interactive API docs URL
	@echo "$(APP_URL)/docs"

# --- Docker ---------------------------------------------------------------
# Uses the same APP_PORT as `make run`, so they cannot both run at once.

up: ## Build and start in Docker (detached)
	docker compose up -d --build

down: ## Stop and remove the Docker containers
	docker compose down

logs: ## Follow the container logs (Ctrl-C exits the view, not the app)
	docker compose logs -f api

ps: ## Show container status
	docker compose ps

rebuild: ## Rebuild the image from scratch, ignoring cache
	docker compose build --no-cache
