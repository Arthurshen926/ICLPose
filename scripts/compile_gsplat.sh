#!/bin/bash
# Pre-compile gsplat JIT with correct CUDA environment
set -e

export CUDA_HOME=/usr/local/cuda-11.6
export PATH=/root/miniconda3/envs/iclpose/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export MAX_JOBS=10

PYTHON=/root/miniconda3/envs/iclpose/bin/python

echo "=== Environment ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $(which nvcc)"
$PYTHON -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"

echo "=== Compiling gsplat CUDA extensions ==="
$PYTHON -c "
from gsplat.cuda._backend import _C
if _C is not None:
    print('SUCCESS: gsplat CUDA backend compiled and loaded')
else:
    print('FAILED: _C is None')
    exit(1)
"
echo "=== Checking cached .so ==="
find /root/.cache/torch_extensions/ -name "*.so" -ls
echo "=== Done ==="
