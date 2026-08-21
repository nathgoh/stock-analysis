.PHONY: help run up down stop restart logs ps status sql-client clean

# Docker compose command
DOCKER_COMPOSE ?= docker compose

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Build and start all containers in the background
	$(DOCKER_COMPOSE) up -d --build

run-news: ### Run the news producer / stream
	uv run python -m news_service

down: ## Stop and remove all containers and networks
	$(DOCKER_COMPOSE) down

stop: ## Stop running containers without removing them
	$(DOCKER_COMPOSE) stop

restart: down run ## Restart all containers

logs: ## Follow logs from all containers
	$(DOCKER_COMPOSE) logs -f

ps: ## Show status of running containers
	$(DOCKER_COMPOSE) ps

status: ps ## Alias for 'ps'

sql-client: ## Open an interactive Flink SQL client session
	$(DOCKER_COMPOSE) run --rm sql-client

clean: ## Stop and remove containers along with volumes
	$(DOCKER_COMPOSE) down -v
