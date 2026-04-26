conda create -n iclpose python=3.9 -y

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt

# Optional (for Gaussian CUDA extensions)
# git submodule update --init --recursive
# pip install submodules/diff-gaussian-rasterization
# pip install submodules/simple-knn
# pip install -U git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0