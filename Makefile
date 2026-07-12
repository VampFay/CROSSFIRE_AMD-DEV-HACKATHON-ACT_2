# ============================================================
# Crossfire — Makefile
# ============================================================
# Common development commands. Run `make help` for available targets.
# ============================================================

.PHONY: help install test lint run docker-up docker-down docker-logs docker-ps \
        rag dataset baselines demo clean

help:
	@echo "Crossfire — Available targets:"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install Python dependencies"
	@echo ""
	@echo "Data preparation:"
	@echo "  make rag              Build RAG corpus (ChromaDB)"
	@echo "  make dataset          Generate training dataset"
	@echo "  make baselines        Generate baseline outputs for CUDA samples"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run backend tests"
	@echo "  make lint             Run linters (ruff, mypy)"
	@echo ""
	@echo "Development:"
	@echo "  make run              Start backend (UI at http://localhost:8000/ui)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        Start all services via docker-compose"
	@echo "  make docker-down      Stop all services"
	@echo "  make docker-logs      Tail logs from all services"
	@echo "  make docker-ps        Show service status"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove build artifacts and caches"

# ============================================================
# Setup
# ============================================================

install:
	cd backend && pip install -r ../requirements.txt

# ============================================================
# Data preparation
# ============================================================

rag:
	cd backend && python3 scripts/build_rag.py

dataset:
	cd backend && python3 scripts/prepare_dataset.py --output data/cuda_rocm_pairs.jsonl

baselines:
	cd backend && python3 scripts/generate_baselines.py

# ============================================================
# Testing
# ============================================================

test:
	cd backend && FIREWORKS_API_KEY=test ALLOW_STUB_FALLBACK=true ALLOW_STUB_SANDBOX=true python3 -m pytest tests/ -v

lint:
	cd backend && ruff check app/ tests/ scripts/ || true
	cd backend && mypy app/ --ignore-missing-imports || true

# ============================================================
# Development
# ============================================================

run:
	cd backend && uvicorn app.main:app --reload --port 8000

# ============================================================
# Docker
# ============================================================

docker-up:
	docker compose up -d
	@echo ""
	@echo "Services starting. Check status with: make docker-ps"
	@echo "UI:       http://localhost:8000/ui"
	@echo "API docs: http://localhost:8000/docs"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

# ============================================================
# Demo
# ============================================================

demo:
	@echo "Running demo verification..."
	curl -sf http://localhost:8000/health | python3 -m json.tool
	@echo ""
	@echo "Testing sample translation..."
	curl -X POST http://localhost:8000/api/translate-sync \
		-H "Content-Type: application/json" \
		-d '{"cuda_source": "__global__ void k(float* x) { x[0] = 1.0f; }", "filename": "test.cu"}' \
		| python3 -m json.tool | head -20

# ============================================================
# Maintenance
# ============================================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/chroma_db backend/data/translation_memory.db
	@echo "Cleaned build artifacts and caches"
