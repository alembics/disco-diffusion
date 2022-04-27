# One-liner while using Docker during development
# Meant to be run from the home directory of this repo
# Purge garbage layers
docker image prune -f && \
# Re-build the docker file
docker build -t disco-diffusion:dev --file ../docker/main/Dockerfile . && \
# Run a test docker command, adding whatever is currently in your disco-diffusion cwd.
# Make note to change the volume mappings to reflect your host env
docker run --rm -it \
    --gpus device=all --name="chris" --ipc=host --user $(id -u):$(id -g) \
    -v /home/mike/ai/disco5/images_out:/workspace/code/images_out \
    -v /home/mike/ai/disco5/init_images:/workspace/code/init_images \
    -v /home/mike/disco-diffusion-1:/workspace/scratch \
    -e xinit_image='"init_images/rn50.png"' \
    -e xskip_steps=70 \
    -e animation_mode='"2D"' \
    -e cuda_device='"cuda:0"' \
    -e simple_nvidia_smi_display='true' \
    -e batch_name='"chris"' \
    -e width_height='[1280,768]' \
    -e xclip_guidance_scale=5000 \
    -e steps=250 \
    -e n_batches=100 \
    -e display_rate=10 \
    -e console_preview=false \
    -e set_seed=69421301 \
    -e RN50=false \
    -e RN50x16=false \
    -e RN50x64=true \
    -e ViTB32=true \
    -e ViTB16=true \
    -e ViTL14_336=false \
    -e ViTL14=false \
    -e cut_ic_pow=100 \
    -e resume_run=true \
    -e run_to_resume=19 \
    -e text_prompts='{"0":["decaying zombie, rotting flesh, tumors and teeth with maggots and pimples, gaping vagina, undead cthulu, trending on artstation, art by lisa wolfe"]}' \
    disco-diffusion:dev bash -c "cp -r /workspace/scratch/* disco-diffusion-1/ && python disco-diffusion-1/disco.py"