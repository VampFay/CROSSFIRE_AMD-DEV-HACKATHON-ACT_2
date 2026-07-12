# ============================================================
# Crossfire Backend Dockerfile
# ============================================================
# Base: ROCm 7.2.4 + PyTorch 3.0 (Ubuntu 24.04, Python 3.11)
# Target: AMD Instinct MI300X (gfx942)

FROM rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.11_pytorch_3.0

LABEL org.opencontainers.image.title="Crossfire"
LABEL org.opencontainers.image.description="Autonomous CUDA-to-ROCm Translation Agent"
LABEL org.opencontainers.image.source="https://github.com/VampFay/CROSSFIRE_AMD-DEV-HACKATHON-ACT_2"
LABEL org.opencontainers.image.licenses="MIT"

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    hip-dev hipblaslt-dev miopen-hip-dev rocprim-dev rocthrust-dev \
    rccl-dev rocfft-dev rocrand-dev rocsolver-dev rocsparse-dev \
    rocalution-dev hipfft-dev hiprand-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY backend/ ./backend/
COPY samples/ ./samples/

WORKDIR /app/backend

# ---- Environment ----
# MI300X is gfx942 — NO HSA_OVERRIDE_GFX_VERSION needed (that's for MI100/MI200)
ENV PYTHONPATH=/app/backend \
    PYTORCH_ROCM_ARCH=gfx942 \
    PYTHONUNBUFFERED=1

# ---- Health check ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
