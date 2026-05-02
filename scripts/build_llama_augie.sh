#!/bin/bash
#SBATCH --job-name=build_llama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sarora01@villanova.edu

# Build llama-cpp-python with CUDA on the compute node (where nvcc + driver exist).
# Run this ONCE before the main Phase 5 job.
set -euo pipefail

echo "Build job $SLURM_JOB_ID on $SLURMD_NODENAME — $(date)"

# Activate venv
source "$HOME/polca_venv/bin/activate"
echo "Python: $(which python3)"

# Show CUDA environment
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version 2>/dev/null || echo "nvcc not in PATH — will use full path"

# Find nvcc
NVCC_PATH=$(which nvcc 2>/dev/null || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -1 || echo "")
if [ -z "$NVCC_PATH" ]; then
    echo "ERROR: nvcc not found on compute node" >&2; exit 1
fi
CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
echo "CUDA_HOME: $CUDA_HOME  nvcc: $NVCC_PATH"

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

pip install -U pip setuptools wheel

CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$NVCC_PATH" \
FORCE_CMAKE=1 \
  pip install --force-reinstall --no-cache-dir "llama-cpp-python>=0.2.90" 2>&1

# Verify
python3 -c "import llama_cpp; print('llama_cpp OK:', llama_cpp.__version__)"

echo "Build complete — $(date)"
echo "You can now submit run_phase5_augie.sh"
