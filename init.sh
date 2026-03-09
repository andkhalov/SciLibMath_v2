#!/usr/bin/env bash
# =============================================================================
# init.sh — SciLibMath_v2 Project Setup
# =============================================================================
# Автоматическая настройка окружения: venv, зависимости, данные, GPU, S3
# Запуск: bash init.sh
# Обновлять при ЛЮБОМ изменении зависимостей или структуры!
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; ((PASS++)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; ((FAIL++)); }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; ((WARN++)); }

echo "============================================"
echo "  SciLibMath_v2 — Project Setup"
echo "  $(date -Iseconds)"
echo "============================================"
echo

# -----------------------------------------------
# 1. Python version
# -----------------------------------------------
echo "--- 1. Python ---"
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
        ok "Python $PY_VER"
    else
        fail "Python $PY_VER (need >= 3.11)"
    fi
else
    fail "python3 not found"
fi

# -----------------------------------------------
# 2. Virtual environment
# -----------------------------------------------
echo "--- 2. Virtual Environment ---"
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating venv..."
    python3 -m venv "$VENV_DIR"
    ok "venv created at $VENV_DIR"
else
    ok "venv exists at $VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
ok "venv activated ($(which python))"

# -----------------------------------------------
# 3. Dependencies
# -----------------------------------------------
echo "--- 3. Dependencies ---"
pip install --upgrade pip -q 2>/dev/null
if pip install -r requirements.txt -q 2>/dev/null; then
    ok "requirements.txt installed"
else
    fail "pip install failed"
fi

# Install dataset package
DATASET_REPO="$SCRIPT_DIR/data/scilibrumodal-v2"
if [ -d "$DATASET_REPO" ]; then
    if pip install -e "$DATASET_REPO" -q 2>/dev/null; then
        ok "scilibrumodal-v2 package installed"
    else
        fail "scilibrumodal-v2 package install failed"
    fi
else
    warn "Dataset repo not cloned yet (run: git clone git@github.com:andkhalov/scilibrumodal-v2.git data/scilibrumodal-v2)"
fi

# -----------------------------------------------
# 4. GPU
# -----------------------------------------------
echo "--- 4. GPU ---"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $GPU_NAME ($GPU_MEM)"

    # Check CUDA in PyTorch
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        ok "PyTorch CUDA: $CUDA_VER"
    else
        fail "PyTorch cannot see CUDA"
    fi
else
    warn "nvidia-smi not found — CPU only"
fi

# -----------------------------------------------
# 5. Dataset
# -----------------------------------------------
echo "--- 5. Dataset ---"
DATA_DIR="$SCRIPT_DIR/data/scilibrumodal-v2-data"
if [ -d "$DATASET_REPO" ]; then
    ok "Dataset repo cloned"
else
    echo "  Cloning dataset repo..."
    mkdir -p "$SCRIPT_DIR/data"
    if git clone git@github.com:andkhalov/scilibrumodal-v2.git "$DATASET_REPO" 2>/dev/null; then
        ok "Dataset repo cloned"
    else
        fail "Failed to clone dataset repo"
    fi
fi

# Check if data is downloaded
if [ -d "$DATA_DIR/train" ] && [ -f "$DATA_DIR/dataset_dict.json" ]; then
    ARROW_COUNT=$(ls "$DATA_DIR/train/"*.arrow 2>/dev/null | wc -l)
    ok "Dataset downloaded ($ARROW_COUNT arrow shards)"
else
    echo "  Downloading dataset from S3..."
    if python -c "
from scilibrumodal_v2_data import ensure_scilibrumodal_v2_local
ensure_scilibrumodal_v2_local('$DATA_DIR')
" 2>/dev/null; then
        ok "Dataset downloaded"
    else
        warn "Auto-download failed. Manual: python -c \"from scilibrumodal_v2_data import ensure_scilibrumodal_v2_local; ensure_scilibrumodal_v2_local('$DATA_DIR')\""
    fi
fi

# Check if images are unpacked
if [ -d "$DATA_DIR/img" ]; then
    IMG_COUNT=$(ls "$DATA_DIR/img/" 2>/dev/null | wc -l)
    if [ "$IMG_COUNT" -gt 0 ]; then
        ok "Images unpacked ($IMG_COUNT files)"
    else
        warn "img/ directory exists but empty"
    fi
else
    if [ -f "$DATA_DIR/img.tar.zst" ]; then
        echo "  Unpacking images (this may take a while)..."
        if python -c "
from scilibrumodal_v2_data import unpack_images
unpack_images('$DATA_DIR')
" 2>/dev/null; then
            ok "Images unpacked"
        else
            warn "Auto-unpack failed. Manual: python -c \"from scilibrumodal_v2_data import unpack_images; unpack_images('$DATA_DIR')\""
        fi
    else
        warn "No img.tar.zst found — images not available"
    fi
fi

# -----------------------------------------------
# 6. Directories
# -----------------------------------------------
echo "--- 6. Directories ---"
for DIR in checkpoints runs logs; do
    mkdir -p "$SCRIPT_DIR/$DIR"
done
ok "checkpoints/, runs/, logs/ exist"

# -----------------------------------------------
# 7. S3 / MinIO access
# -----------------------------------------------
echo "--- 7. S3 Backup ---"
if command -v rclone &>/dev/null; then
    ok "rclone installed ($(rclone --version 2>/dev/null | head -1))"
    # Test MinIO access if configured
    if rclone lsd minio: 2>/dev/null | head -1 >/dev/null; then
        ok "MinIO accessible via rclone"
    else
        warn "MinIO not configured in rclone. Setup: rclone config"
    fi
elif command -v rsync &>/dev/null; then
    ok "rsync available (fallback for S3 backup)"
else
    warn "Neither rclone nor rsync found — S3 backup disabled"
fi

# -----------------------------------------------
# 8. Smoke test
# -----------------------------------------------
echo "--- 8. Smoke Test ---"
if python -c "
import torch
import datasets
import numpy as np
from pathlib import Path
print(f'  torch={torch.__version__}, datasets={datasets.__version__}, numpy={np.__version__}')
" 2>/dev/null; then
    ok "Core imports successful"
else
    fail "Core imports failed"
fi

# Test dataset loading (if data exists)
if [ -d "$DATA_DIR/train" ]; then
    if python -c "
from scilibrumodal_v2_data import load_scilibrumodal_v2
dsd = load_scilibrumodal_v2('$DATA_DIR', normalize=True)
ds = dsd['train']
print(f'  Dataset: {len(ds)} rows, columns: {ds.column_names}')
sample = ds[0]
print(f'  Sample keys: {list(sample.keys())[:5]}...')
" 2>/dev/null; then
        ok "Dataset loads successfully"
    else
        fail "Dataset loading failed"
    fi
else
    warn "Dataset not downloaded — skipping load test"
fi

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo
echo "============================================"
echo "  Summary: ${GREEN}$PASS OK${NC}, ${RED}$FAIL FAIL${NC}, ${YELLOW}$WARN WARN${NC}"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Fix failures before starting experiments.${NC}"
    exit 1
else
    echo -e "${GREEN}Ready to train!${NC}"
    echo "  Activate: source venv/bin/activate"
    echo "  Train:    python code/train.py --config configs/e1_pairwise.yaml"
    exit 0
fi
