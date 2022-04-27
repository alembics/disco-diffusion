# One-liner while using Docker during development
# Meant to be run from the home directory of this repo
# Purge garbage layers
docker image prune -f && \
# Re-build the docker file
docker build -t disco-diffusion:dev --file ../docker/main/Dockerfile . && \
# Run a test docker command, adding whatever is currently in your disco-diffusion cwd.
# Make note to change the volume mappings to reflect your host env
docker run --rm -it \
    --gpus device=all --name="galaxy" --ipc=host --user $(id -u):$(id -g) \
    -v /home/mike/ai/disco5/images_out:/workspace/code/images_out \
    -v /home/mike/ai/disco5/init_images:/workspace/code/init_images \
    -v /home/mike/disco-diffusion-1:/workspace/scratch \
    -e cuda_device='"cuda:0"' \
    -e simple_nvidia_smi_display='true' \
    -e batch_name='"galaxy"' \
    -e width_height='[1280,768]' \
    -e xclip_guidance_scale=5000 \
    -e xwidth_height='[768,1024]' \
    -e xinit_image='"init_images/samyang.jpg"' \
    -e xskip_steps=100 \
    -e steps=50 \
    -e n_batches=100 \
    -e display_rate=10 \
    -e console_preview=false \
    -e set_seed=69421301 \
    -e RN50=true \
    -e RN50x16=false \
    -e RN50x64=false \
    -e ViTB32=true \
    -e ViTB16=true \
    -e ViTL14_336=true \
    -e ViTL14=false \
    -e cut_ic_pow=100 \
    -e text_prompts='{"0":["woman with hydrangea hair, trending on artstation, digital art by sam yang"]}' \
    disco-diffusion:dev bash -c "cp -r /workspace/scratch/* disco-diffusion-1/ && python disco-diffusion-1/disco.py"