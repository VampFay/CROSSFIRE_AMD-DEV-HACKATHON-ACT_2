# Crossfire — lablab.ai Submission Checklist

> All 13 submission fields for the AMD Developer Hackathon: ACT II.
> Copy-paste from this document into the lablab.ai submission form.

---

## Track Selection

**Best AMD-Hosted Gemma Project** — Crossfire runs Gemma 4 12B (`unsloth/gemma-4-12b-it`) via vLLM on AMD MI300X (gfx942, ROCm 7.2.4) as the local translation model. Hard semantic cases fall back to Gemma 27B via Fireworks AI. Every translation is compiled, run, and validated on AMD hardware.

---

## Field 1: Project Title
```
Crossfire: Autonomous CUDA-to-ROCm Translation Agent
```

---

## Field 2: Short Description (50 words max)
```
Crossfire is an autonomous AI agent that ports CUDA code to AMD ROCm in minutes, not weeks. It uses HIPIFY-first translation plus LLM semantic repair, compiles on AMD MI300X via ROCm 7.2.4, runs translated code, and numerically validates output against analytically computed expected outputs — fully automated.
```

---

## Field 3: Long Description (500+ words)

NVIDIA's CUDA has been the dominant software moat holding AMD back from AI GPU market share for over 15 years. There are tens of thousands of ML codebases — PyTorch extensions, custom kernels, training scripts, inference pipelines — that teams want to run on cheaper, more available AMD MI300X hardware but cannot without manual rewrite. AMD's own HIPIFY tool performs pure syntax translation (approximately 60 percent of the work) but breaks on semantic patterns: cuDNN-to-MIOpen mapping, warp shuffle primitives, custom kernel logic, and Triton kernels. Every ML team's number one objection to AMD adoption is the same: "We cannot afford to rewrite our CUDA code."

Crossfire closes this gap. It is an autonomous AI agent built on a LangGraph state machine that takes a CUDA source file as input and produces a validated, compiled, and benchmarked ROCm equivalent running on AMD MI300X GPUs. The agent loop runs through six states: static analysis (identifies kernels, cuBLAS, cuDNN, Thrust, Triton, warp shuffle patterns and computes a difficulty score), translation (HIPIFY-first: AMD's `hipify-clang` performs the deterministic syntax pass, then a local Gemma 4 12B served via vLLM on AMD MI300X — or a remote Gemma 27B via Fireworks AI for hard semantic cases — performs only the semantic repair on top of the HIPIFY output), compilation (hipcc in an isolated Docker sandbox with ROCm 7.2.4), execution (runs the translated binary on AMD MI300X with sample inputs), numerical diff (compares outputs against analytically computed expected outputs with a 1e-5 threshold — these are hand-coded predicted values, not captures from NVIDIA hardware; CUDA-on-NVIDIA differential verification is on the roadmap), and debug (formats error feedback for the next iteration). The loop retries until correctness is achieved or a JobBudget (hard caps on iterations, model calls, tokens, cost, wall time) is exhausted.

Every architectural component depends on AMD hardware by design. The agent orchestrator runs on AMD Developer Cloud. Translations are validated by compiling and running on ROCm 7.2.4 with MI300X GPUs (gfx942 architecture). The local translation model is Gemma 4 12B served via vLLM on MI300X — a prompt-engineering baseline. Hard semantic cases fall back to Gemma 27B via the Fireworks AI API, which hosts models on AMD hardware. The RAG layer uses ChromaDB with curated ROCm 7.2.4 documentation chunks, MIOpen API references, and translation tables to augment translation prompts with relevant context.

The project includes 20 CUDA sample programs spanning difficulty levels from simple vector_add through flash-attention subsets and warp-shuffle reductions. Each sample emits structured JSON output between markers that the sandbox parses and numerically diffs against analytically computed baselines. Every result also carries a `VerificationLevel` enum (`analyzed → translated → compiled → executed → test_verified → differentially_verified → benchmarked`) shown as a badge in the UI, plus a GPU attestation block (`gpu_model`, `architecture` (gfx942), `rocm_version`, `hipcc_version`, `compiler_flags`) recording which hardware ran the code. The UI is a standalone web app served by the backend at `/ui/`, with a CUDA/ROCm diff view, live WebSocket status updates, a correctness report card showing max_abs_error and MSE, and a performance comparison chart. The entire stack is containerized with Docker Compose, including the ROCm sandbox, vLLM server, Redis, ChromaDB, FastAPI backend, UI, and a background RQ worker.

Crossfire addresses a real problem: porting CUDA to ROCm is slow and error-prone. AMD lists "porting CUDA workloads to AMD hardware" as a ROCm use case on their hackathon page. The company has acquired Mipsology, Nod.ai, and Silo AI to improve their software stack. Crossfire fits this direction.

Built in five days for the AMD Developer Hackathon ACT II (Track 3 Unicorn Track), Crossfire demonstrates that the compile-validate-iterate loop — a compile-validate-iterate loop that goes beyond syntax translation — can work end-to-end on real AMD hardware. The v0.3.0 submission ships with a prompt-engineering baseline on Gemma 4 12B; v0.4 will expand the CUDA→ROCm dataset, add CUDA-on-NVIDIA differential verification, and extend multi-file repository support.

---

## Field 4: Technology and Category Tags
```
amd-rocm, fireworks-ai, gemma, langgraph, vllm, code-translation, ai-agents, rocm, mi300x, fastapi, docker, chromadb, rag
```

---

## Field 5: Cover Image
- **File**: `assets/cover_image.png`
- **Dimensions**: 1920 x 1080
- **Format**: PNG
- **Size**: ~141 KB
- **Upload**: Upload the file directly to lablab.ai

---

## Field 6: Video Presentation
- **File**: `download/crossfire_gemma4_demo.mp4` (and `.webm` mirror)
- **Duration**: ~18 seconds
- **Format**: MP4, 1080p
- **Script**: See `docs/demo_video_script.md`
- **Upload**: Upload to YouTube as unlisted, paste URL to lablab.ai
- **YouTube URL**: Add the URL here after uploading — the demo file is ready in `download/`.

---

## Field 7: Slide Presentation
- **File**: `docs/pitch_deck.pdf`
- **Source**: `docs/pitch_deck.html`
- **Pages**: 11 slides
- **Format**: PDF
- **Size**: ~476 KB
- **Upload**: Upload the PDF directly to lablab.ai

---

## Field 8: Public GitHub Repository
- **URL**: `https://github.com/VampFay/CROSSFIRE_AMD-DEV-HACKATHON-ACT_2`
- **Requirements**:
  - [x] Public repo
  - [x] README.md with setup and usage instructions
  - [x] MIT LICENSE
  - [x] Runnable with provided instructions (`docker compose up -d`)
  - [x] Containerized (Dockerfile + docker-compose.yml)
  - [x] Tag v0.3.0

---

## Field 9: Demo Application Platform
- **Platform**: AMD Developer Cloud (MI300X instance)
- **Container**: Docker (via docker-compose)
- **Services**: api (FastAPI + UI), vllm, sandbox (ROCm 7.2.4), redis, chromadb, worker

---

## Field 10: Application URL
- **URL**: `http://129.212.185.42:8000/ui/`
- **API docs**: `http://129.212.185.42:8000/docs`
- **Health check**: `http://129.212.185.42:8000/health`
- **Auth**: Basic auth (user: see `.env.example`, pass: see `.env.example`) — protects demo from burning API credits during judging.
- **Note**: URL is live for the duration of judging.

---

## Field 11: Team Members
- **Member 1**: Solo developer — Tech Lead / ML Engineer / Full-stack
  - GitHub: VampFay
  - Role: Agent architecture, HIPIFY-first translation pipeline, RAG, FastAPI backend, UI, Docker/ROCm sandbox, demo video, submission docs

> Single-member team. (If additional collaborators are added before submission, list them here with their GitHub handles and roles.)

---

## Field 12: Hackathon-Specific Requirements Checklist

### Containerized
- [x] Dockerfile present at repo root
- [x] docker-compose.yml with all services
- [x] `docker compose up -d` starts the full stack

### MIT-Compliant License
- [x] LICENSE file is MIT
- [x] All third-party deps are MIT-compatible:
  - PyTorch (BSD) ✓
  - LangGraph (MIT) ✓
  - FastAPI (MIT) ✓
  - Gemma 4 12B (Apache 2.0) ✓
  - ROCm (AMD License / MIT) ✓
  - vLLM (Apache 2.0) ✓
  - ChromaDB (Apache 2.0) ✓
  - hipify-clang (MIT / UIUC) ✓

### Runnable with Provided Instructions
- [x] README has Quick Start section
- [x] `cp .env.example .env` documented
- [x] `docker compose up -d` documented
- [x] Demo URL documented (`http://129.212.185.42:8000/ui/`)
- [x] Tested from clean clone (`langgraph-checkpoint-sqlite` added to `requirements.txt`)

### Original Work
- [x] All code written during hackathon window (July 6-11)
- [x] No GPL code

---

## Field 13: Additional Notes (optional)

### Best AMD-Hosted Gemma Project
Crossfire uses Gemma 4 12B (`unsloth/gemma-4-12b-it`) as the local translation model served via vLLM on AMD MI300X (BF16, gfx942). This is a prompt-engineering baseline. Hard semantic translation cases use Gemma 27B via Fireworks AI. The RAG layer augments prompts with ROCm 7.2.4 documentation to improve translation accuracy. Local tokens are free per the hackathon rules.

### Innovation Highlights
1. **HIPIFY-first, LLM-second.** `hipify-clang` does the deterministic syntax pass; the LLM only repairs semantic gaps (cuDNN→MIOpen, warp shuffle, Triton). More auditable than a pure-LLM approach, and keeps the LLM focused on what it's good at.
2. **Compile-validate-iterate loop.** No existing tool (HIPIFY, Copilot, manual) validates translations by actually running them on AMD hardware. This is what makes it different from HIPIFY alone.
3. **VerificationLevel + GPU Attestation.** Every result carries a verification-level enum and a hardware attestation block (gpu_model, gfx942, rocm_version, hipcc_version, compiler_flags) — reviewers can see exactly how far each translation was validated and on what hardware.
4. **JobBudget.** Hard caps on iterations, model calls, tokens, cost, and wall time prevent infinite loops and unbounded spend.
5. **Migration Engine.** Repository-level migration planning — inventory, build-system detection, capability matrix, risk items, and per-file patch generation — not just single-file translation.
6. **Security hardening.** Filename sanitization (prevents command injection + path traversal); stub fallback defaults to `FALSE` so the system fails closed.
7. **AMD-mandatory architecture.** Every component depends on AMD infrastructure.

### Demo Notes for Judges
- The demo URL has basic auth (user: see `.env.example`, pass: see `.env.example`) to prevent credit burn.
- All hero demos have been tested and pass validation.
- `rocm-smi` output is visible in the demo video to prove real AMD hardware execution.
- The agent loop takes 30–90 seconds per translation depending on difficulty (bounded by JobBudget).
- Each result's VerificationLevel badge shows how far validation proceeded; GPU attestation shows the exact gfx942 / ROCm 7.2.4 / hipcc versions used.

---

## Pre-Submission Verification (Day 6 Morning)

Run this checklist before clicking "Submit" on lablab.ai:

- [ ] Demo URL (`http://129.212.185.42:8000/ui/`) loads in incognito browser (no auth cached)
- [ ] Demo URL works from mobile network (not just localhost)
- [ ] vector_add.cu demo runs end-to-end successfully
- [ ] GitHub repo is PUBLIC (not private)
- [ ] GitHub repo has v0.3.0 tag
- [ ] GitHub README renders correctly (no broken images)
- [ ] YouTube video is unlisted (not private) and link works
- [ ] YouTube video has captions
- [ ] Slide deck PDF opens and all 11 slides render
- [ ] Cover image is 1920x1080 PNG under 2 MB
- [ ] Long description has no typos (run spellcheck)
- [ ] All tech tags are valid lablab.ai tags
- [ ] All team members added as collaborators on lablab.ai
- [ ] `docker build .` succeeds from clean clone
- [ ] No `.env` file in repo (only `.env.example`)
- [ ] No API keys in code or git history
