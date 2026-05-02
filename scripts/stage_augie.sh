#!/usr/bin/env bash
# stage_augie.sh — run this on your LAPTOP to push code + install deps on the
# Augie login node before submitting the SLURM job.
#
# Usage:
#   bash scripts/stage_augie.sh [--skip-models] [--skip-install]
#
# What it does:
#   1. rsync the repo to ~/creative-ai/ on augie (excluding venv, models, outputs)
#   2. Install llama-cpp-python (CUDA) + repo deps into ~/creative-ai-venv/ on the login node
#   3. Download all 6 Phase-5 GGUFs into ~/creative-ai-models/ on the login node
#      (uses huggingface_hub — runs on the login node which HAS internet)
#   4. Writes ~/creative-ai/model_paths.augie.json pointing at the staged paths
#
# Requires: ssh access to augie.villanova.edu with VPN connected.
#
set -euo pipefail

AUGIE="sarora01@augie.villanova.edu"
REMOTE_REPO="~/creative-ai"
REMOTE_MODELS="~/creative-ai-models"
REMOTE_VENV="~/creative-ai-venv"

SKIP_MODELS=false
SKIP_INSTALL=false

for arg in "$@"; do
  case "$arg" in
    --skip-models)   SKIP_MODELS=true  ;;
    --skip-install)  SKIP_INSTALL=true ;;
  esac
done

echo "=== [1/4] Syncing repo to $AUGIE:$REMOTE_REPO ==="
rsync -av --exclude='.git' \
          --exclude='__pycache__' \
          --exclude='.venv' \
          --exclude='.pytest_cache' \
          --exclude='build/' \
          --exclude='*.egg-info' \
          --exclude='models/' \
          --exclude='outputs/' \
          ./ "${AUGIE}:${REMOTE_REPO}/"

echo ""
echo "=== [2/4] Installing deps on Augie login node ==="
if [ "$SKIP_INSTALL" = true ]; then
  echo "  (skipped via --skip-install)"
else
  ssh "$AUGIE" bash <<'REMOTE_INSTALL'
set -euo pipefail
source "$HOME/polca_venv/bin/activate" 2>/dev/null || {
  echo "polca_venv not found — creating creative-ai-venv"
  module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
  python3 -m venv ~/creative-ai-venv
  source ~/creative-ai-venv/bin/activate
}

echo "Python: $(which python3)"

# Install llama-cpp-python with CUDA.
# Strategy: try pre-built cu121 wheel first (binary-compatible with CUDA 13.x at runtime),
# then try cu122/cu123, then fall back to source build with explicit nvcc path.
pip install -U pip setuptools wheel

LLAMA_INSTALLED=false

for cuda_tag in cu121 cu122 cu123; do
    echo "  Trying pre-built wheel: llama-cpp-python[$cuda_tag] ..."
    pip install "llama-cpp-python>=0.2.90" \
        --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${cuda_tag}" \
        --force-reinstall --no-cache-dir 2>&1 | tail -5 && LLAMA_INSTALLED=true && break || true
done

if [ "$LLAMA_INSTALLED" = false ]; then
    echo "  Pre-built wheels failed — building from source with nvcc."
    CUDA_HOME_GUESS=""
    for candidate in /usr/local/cuda-13.1 /usr/local/cuda-13 /usr/local/cuda-12.1 /usr/local/cuda; do
        if [ -x "$candidate/bin/nvcc" ]; then
            CUDA_HOME_GUESS="$candidate"; break
        fi
    done
    [ -z "$CUDA_HOME_GUESS" ] && CUDA_HOME_GUESS="/usr/local/cuda"
    echo "  Using CUDA_HOME: $CUDA_HOME_GUESS"
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=${CUDA_HOME_GUESS}/bin/nvcc" FORCE_CMAKE=1 \
        pip install --force-reinstall --no-cache-dir "llama-cpp-python>=0.2.90"
fi
echo "  llama-cpp-python installed."

# Install the creativeai package + data/HF extras
pip install -e ~/creative-ai/.[data]
pip install huggingface_hub

echo "Install complete."
REMOTE_INSTALL
fi

echo ""
echo "=== [3/4] Downloading Phase-5 GGUFs to Augie login node ==="
if [ "$SKIP_MODELS" = true ]; then
  echo "  (skipped via --skip-models)"
else
  # Write a download helper script to Augie and run it — avoids here-doc quoting issues.
  cat > /tmp/_creativeai_download.py << 'PYEOF'
import json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
from huggingface_hub import hf_hub_download

out_dir = Path.home() / "creative-ai-models"
out_dir.mkdir(parents=True, exist_ok=True)

models = {
    "gemma-2b":                 ("llm-exp/gemma-2b-Q4_K_M-GGUF",             "gemma-2b.Q4_K_M.gguf"),
    "gemma-2b-it":              ("reach-vb/gemma-2b-it-Q4_K_M-GGUF",         "gemma-2b-it.Q4_K_M.gguf"),
    "qwen2.5-3b":               ("Qwen/Qwen2.5-3B-GGUF",                      "qwen2.5-3b-q4_k_m.gguf"),
    "qwen2.5-3b-instruct":      ("Qwen/Qwen2.5-3B-Instruct-GGUF",            "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "mistral-7b-v0.3":          ("QuantFactory/Mistral-7B-v0.3-GGUF",         "Mistral-7B-v0.3.Q4_K_M.gguf"),
    "mistral-7b-instruct-v0.3": ("bartowski/Mistral-7B-Instruct-v0.3-GGUF",  "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
}

paths = {}
for model_id, (repo_id, filename) in models.items():
    dest = out_dir / filename
    if dest.exists():
        print(f"  already present: {filename} ({dest.stat().st_size // 1_000_000} MB)", flush=True)
        paths[model_id] = str(dest.resolve())
    else:
        print(f"  downloading {model_id}: {repo_id}/{filename} ...", flush=True)
        p = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(out_dir))
        paths[model_id] = str(Path(p).resolve())
        print(f"    -> {paths[model_id]}", flush=True)

map_path = Path.home() / "creative-ai" / "model_paths.augie.json"
map_path.write_text(json.dumps(paths, indent=2))
print(f"\nModel path map written to: {map_path}", flush=True)
PYEOF

  scp /tmp/_creativeai_download.py "${AUGIE}:/tmp/_creativeai_download.py"
  ssh "$AUGIE" bash -c '
    source ~/creative-ai-venv/bin/activate 2>/dev/null || source ~/polca_venv/bin/activate 2>/dev/null || true
    HF_HUB_DISABLE_XET=1 python3 /tmp/_creativeai_download.py
  '
fi

echo ""
echo "=== [4/4] Making SLURM script executable on Augie ==="
ssh "$AUGIE" "chmod +x ${REMOTE_REPO}/scripts/run_phase5_augie.sh"

echo ""
echo "=== Staging complete. ==="
echo ""
echo "Submit the job with:"
echo "  ssh $AUGIE \"cd ${REMOTE_REPO} && sbatch scripts/run_phase5_augie.sh\""
