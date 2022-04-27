# One-liner while using Docker during development
# Meant to be run from the home directory of this repo
# Purge garbage layers
docker image prune -f && \
# Re-build the docker file
docker build -t disco-diffusion:dev --file ../docker/main/Dockerfile . && \
# Run a test docker command, adding whatever is currently in your disco-diffusion cwd.
# Make note to change the volume mappings to reflect your host env
docker run --rm -it \
    --gpus device=all --cpus=2.0 --name="horror2" --ipc=host --user $(id -u):$(id -g) \
    -v /home/mike/ai/disco5/images_out:/workspace/code/images_out \
    -v /home/mike/ai/disco5/init_images:/workspace/code/init_images \
    -v /home/mike/disco-diffusion-1:/workspace/scratch \
    -e xinit_image='"init_images/rn50.png"' \
    -e xskip_steps=70 \
    -e cuda_device='"cuda:0"' \
    -e simple_nvidia_smi_display='false' \
    -e batch_name='"horror2"' \
    -e width_height='[1280,768]' \
    -e clip_guidance_scale=500 \
    -e steps=275 \
    -e n_batches=100 \
    -e display_rate=10 \
    -e console_preview=true \
    -e console_preview_width=120 \
    -e set_seed=86753092 \
    -e RN50=true \
    -e RN50x16=true \
    -e RN50x64=false \
    -e ViTB32=true \
    -e ViTB16=true \
    -e ViTL14_336=true \
    -e ViTL14=false \
    -e cut_ic_pow=80 \
    -e text_prompts='{"0":["aerial view of a graphic ground battle, on acid, neorealistic psychedelic journey"]}' \
    disco-diffusion:dev bash -c "cp -r /workspace/scratch/* disco-diffusion-1/ && python disco-diffusion-1/disco.py"