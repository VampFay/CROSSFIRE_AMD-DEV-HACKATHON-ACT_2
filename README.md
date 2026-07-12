# Crossfire

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROCm 7.2.4](https://img.shields.io/badge/ROCm-7.2.4-red.svg)](https://rocm.docs.amd.com/)
[![AMD MI300X](https://img.shields.io/badge/Built%20on-AMD%20MI300X-red.svg)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

Crossfire translates CUDA source files to AMD ROCm and validates the result by compiling and running them on MI300X. It uses AMD's `hipify-clang` for the initial syntax conversion, then uses a Gemma 4 12B model to fix anything hipify can't handle. Each translation is compiled with `hipcc`, run with sample inputs, and checked against expected outputs.

Built for the AMD Developer Hackathon: ACT II. Live demo: http://129.212.185.42:8000/ui/

---

## Quick Start

```bash
git clone https://github.com/VampFay/CROSSFIRE_AMD-DEV-HACKATHON-ACT_2.git
cd CROSSFIRE_AMD-DEV-HACKATHON-ACT_2
cp .env.example .env
docker compose up -d
```

- Web UI: http://localhost:8000/ui/
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

Without Docker:

```bash
pip install -r requirements.txt
cd backend && uvicorn app.main:app --reload --port 8000
```

---

## How It Works

```
CUDA code
    │
    ▼
Analyze ──▶ Translate ──▶ Compile ──▶ Run ──▶ Diff ──▶ Done
               │                                    │
               │  1. hipify-clang (syntax)          │
               │  2. Gemma 4 12B (semantic repair)  │
               │                                    │
               └──── Debug ◀──── retry on fail ─────┘
```

1. **Analyze** — Static analysis detects CUDA patterns (kernels, cuBLAS, shared memory, etc.)
2. **Translate** — hipify-clang does the deterministic pass. If it can't handle something, Gemma 4 12B fills the gaps.
3. **Compile** — `hipcc` compiles the translated HIP code on MI300X.
4. **Run** — The compiled binary runs with sample inputs on the GPU.
5. **Diff** — Output is compared against expected values. If the error is too high, it loops back to debug.
6. **Done** — Result includes a verification badge and GPU attestation.

---

## Tech Stack

| Component | What we use |
|-----------|-------------|
| GPU | AMD MI300X (gfx942) |
| ROCm | 7.2.4 |
| Translation | hipify-clang + Gemma 4 12B |
| Model serving | vLLM |
| Agent framework | LangGraph |
| Backend | FastAPI |
| UI | Standalone HTML at `/ui` |

---

## Project Structure

```
backend/app/
├── agents/       # LangGraph state machine (6 nodes)
├── analyzers/    # CUDA pattern detection
├── models/       # vLLM + Fireworks clients
├── sandbox/      # hipcc compile + run + diff
├── migration/    # hipify adapter + repo planner
├── memory/       # Translation cache (SQLite)
├── rag/          # ROCm docs retrieval
├── routers/      # API endpoints
└── ui/           # Web UI (index.html)

samples/cuda/     # 20 CUDA test programs
samples/baselines/ # Expected outputs for validation
demo/             # Demo video
docs/             # Pitch deck + submission docs
```

---

## API

| Method | Path | What it does |
|--------|------|--------------|
| POST | `/api/translate` | Translate a CUDA file (async) |
| POST | `/api/translate-sync` | Translate a CUDA file (blocking) |
| GET | `/api/jobs/{id}` | Get result with verification badge |
| GET | `/api/samples` | List CUDA samples |
| GET | `/health` | Health check |
| GET | `/ui/` | Web UI |

---

## Testing

```bash
cd backend && python -m pytest tests/ -v
```

77 tests cover CUDA pattern detection, translation, routing, numerical diff, API routes, and filename sanitization.

---

## Limitations

- **No fine-tuning.** We use Gemma 4 12B with prompt engineering. A training pipeline exists but hasn't been run (needs 1000+ training pairs, we have ~23).
- **Analytical baselines.** Outputs are checked against hand-coded expected values, not against actual NVIDIA GPU runs.
- **cuDNN and Triton.** Best-effort support; manual review recommended.
- **Multi-file repos.** Supported but each file is translated independently — no cross-file dependency resolution yet.

---

## License

MIT
