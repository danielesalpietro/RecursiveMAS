COMPOSE := docker compose -f docker-compose.yml

.DEFAULT_GOAL := help

.PHONY: help env core data mlops ai ui security all \
        down clean logs ps pull-models status

help:
	@echo ""
	@echo "  Enterprise GenAI Platform"
	@echo "  ═══════════════════════════════════════════════════════"
	@echo ""
	@echo "  Setup"
	@echo "  ────────────────────────────────────────────────────────"
	@echo "  make env          Copy .env.example → .env"
	@echo ""
	@echo "  Start (cumulative layers)"
	@echo "  ────────────────────────────────────────────────────────"
	@echo "  make core         Postgres + Redis + MinIO"
	@echo "  make data         + Airflow + Spark + Livy"
	@echo "  make mlops        + MLflow + Qdrant"
	@echo "  make ai           + Ollama + RAG API + Mem0"
	@echo "  make ui           + Open WebUI + Superset"
	@echo "  make security     + Keycloak + Traefik"
	@echo "  make all          Everything at once"
	@echo ""
	@echo "  Operations"
	@echo "  ────────────────────────────────────────────────────────"
	@echo "  make pull-models  Pull LLM + embedding models into Ollama"
	@echo "  make status       Show all service health"
	@echo "  make logs         Tail all logs"
	@echo "  make ps           List running containers"
	@echo "  make down         Stop everything (keep volumes)"
	@echo "  make clean        Stop + delete all volumes  ⚠  DESTRUCTIVE"
	@echo ""
	@echo "  Service URLs (defaults)"
	@echo "  ────────────────────────────────────────────────────────"
	@echo "  Open WebUI   → http://localhost:3000"
	@echo "  Superset     → http://localhost:8088"
	@echo "  Airflow      → http://localhost:8080"
	@echo "  MLflow       → http://localhost:5000"
	@echo "  Keycloak     → http://localhost:8443"
	@echo "  MinIO        → http://localhost:9001"
	@echo "  Spark UI     → http://localhost:8082"
	@echo "  Traefik      → http://localhost:8090"
	@echo "  RAG API      → http://localhost:8000/docs"
	@echo "  Mem0 API     → http://localhost:8001/docs"
	@echo ""

env:
	@if [ ! -f .env ]; then cp .env.example .env && echo ".env created — edit before starting"; \
	else echo ".env already exists"; fi

core: env
	$(COMPOSE) up -d postgres redis minio
	@echo "Core infrastructure started."

data: core
	$(COMPOSE) --profile data up -d
	@echo "Data layer started. Airflow UI: http://localhost:8080"

mlops: core
	$(COMPOSE) --profile mlops up -d
	@echo "MLOps layer started. MLflow: http://localhost:5000  Qdrant: http://localhost:6333"

ai: mlops
	$(COMPOSE) --profile ai up -d
	@echo "AI layer started. Run 'make pull-models' to download LLM models."

ui: core
	$(COMPOSE) --profile ui up -d
	@echo "UI layer started. Open WebUI: http://localhost:3000  Superset: http://localhost:8088"

security: core
	$(COMPOSE) --profile security up -d
	@echo "Security layer started. Keycloak: http://localhost:8443"

all: env
	$(COMPOSE) --profile data --profile mlops --profile ai --profile ui --profile security up -d
	@echo "All services started. See 'make help' for URLs."

pull-models:
	@echo "Pulling Ollama models (this may take several minutes)..."
	docker exec $$(docker ps -qf "name=enterprise-ollama-1") ollama pull $${LLM_MODEL:-llama3.2}
	docker exec $$(docker ps -qf "name=enterprise-ollama-1") ollama pull $${EMBEDDING_MODEL:-nomic-embed-text}
	@echo "Models ready."

status:
	$(COMPOSE) --profile data --profile mlops --profile ai --profile ui --profile security ps

logs:
	$(COMPOSE) --profile data --profile mlops --profile ai --profile ui --profile security logs -f

ps:
	$(COMPOSE) ps

down:
	$(COMPOSE) --profile data --profile mlops --profile ai --profile ui --profile security down

clean:
	@echo "⚠  This will DELETE all data volumes. Press Ctrl+C to abort..."
	@sleep 5
	$(COMPOSE) --profile data --profile mlops --profile ai --profile ui --profile security down -v --remove-orphans
