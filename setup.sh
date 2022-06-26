# Install dependencies
conda run -n diffusion pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/openai/CLIP
git clone https://github.com/kostarion/guided-diffusion
git clone https://github.com/assafshocher/ResizeRight.git
git clone https://github.com/MSFTserver/pytorch3d-lite.git
git clone https://github.com/isl-org/MiDaS.git
conda run -n diffusion pip install -e ./CLIP
conda run -n diffusion pip install -e ./guided-diffusion

conda run -n diffusion pip install lpips IPython requests timm einops omegaconf
conda install -y jupyter opencv pandas numpy matplotlib

wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -P models
