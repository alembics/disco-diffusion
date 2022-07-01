############################
# for disco diffusion
############################

# Setup virtual environment
conda run -n diffusion pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/openai/CLIP
git clone https://github.com/kostarion/guided-diffusion
git clone https://github.com/princeton-vl/RAFT
git clone https://github.com/assafshocher/ResizeRight.git
git clone https://github.com/MSFTserver/pytorch3d-lite.git
git clone https://github.com/isl-org/MiDaS.git
git clone https://github.com/shariqfarooq123/AdaBins.git
mkdir pretrained
wget https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt -P pretrained
conda run -n diffusion pip install -e ./CLIP
conda run -n diffusion pip install -e ./guided-diffusion

conda run -n diffusion pip install lpips IPython requests timm einops omegaconf imutils
conda install -n diffusion -y jupyter ipykernel opencv pandas numpy matplotlib scikit-image

# Install models
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -P models
wget https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt -P models
wget https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth -P models

############################
# For diffusion inpainting
############################

git clone https://github.com/discus0434/latent-diffusion.git
cd latent-diffusion

# Setup virtual environment
<<<<<<< HEAD

# Install models
conda env create -f environment.yaml
wget -O latent-diffusion/models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1

=======
>>>>>>> 4d3397b014f89e162128e299afe1d59c95211a42

# Install models
conda env create -f environment.yaml
wget -O latent-diffusion/models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
