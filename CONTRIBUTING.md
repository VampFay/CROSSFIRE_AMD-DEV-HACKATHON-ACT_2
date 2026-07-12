# Contributing to Crossfire

Thanks for your interest in contributing to Crossfire! This document covers the development workflow.

## Development Setup

### Prerequisites

- Python 3.10+
- Docker 24+ with Docker Compose v2
- (Optional) AMD GPU with ROCm 7.2.3 support

### Quick Start

```bash
git clone https://github.com/VampFay/CROSSFIRE_AMD-DEV-HACKATHON-ACT_2.git
cd CROSSFIRE_AMD-DEV-HACKATHON-ACT_2

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
cp ../.env.example ../.env
# Edit .env: set FIREWORKS_API_KEY if using remote model

# Run (UI at http://localhost:8000/ui)
uvicorn app.main:app --reload --port 8000
```

## Development Workflow

### 1. Create a branch

```bash
git checkout -b feature/my-feature
```

### 2. Make changes

Follow the existing code style. Run linters before committing:

```bash
cd backend
ruff check .
mypy app/
```

### 3. Run tests

```bash
cd backend
pytest tests/ -v
```

### 4. Commit

Use conventional commits:

```
feat: add cuDNN→MIOpen translation pattern
fix: handle empty CUDA source in analyzer
docs: update README with deployment instructions
```

### 5. Open a PR

Push your branch and open a Pull Request. CI will run automatically.

## Code Style

### Python

- Follow PEP 8 (ruff enforces this)
- Type hints required on all function signatures
- Docstrings on all public functions (Google style)
- Max line length: 100 characters
- Use `from __future__ import annotations` for forward references

### CUDA Samples

Each sample in `samples/cuda/` must:
- Have a header comment with filename, description, and difficulty rating
- Include `<cuda_runtime.h>` and any library headers
- Print outputs in JSON format between `===OUTPUT_BEGIN===` and `===OUTPUT_END===` markers
- Verify results against a CPU reference implementation
- Print `max_error` and `status` (pass/fail) in the output JSON

## Testing

### Unit Tests

Located in `backend/tests/`. Run with:

```bash
cd backend && pytest tests/ -v
```

### Integration Tests

Manual integration tests via the API:

```bash
curl -X POST http://localhost:8000/api/translate-sync \
  -H "Content-Type: application/json" \
  -d '{"cuda_source": "__global__ void k() {} cudaMalloc(&p, 1024);"}'
```

### End-to-End Tests

Requires full Docker stack running:

```bash
docker compose up -d
# Wait for vllm to be healthy (~2 min)
# Open http://localhost:8000/ui and test manually
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Built for the AMD Developer Hackathon: ACT II.
