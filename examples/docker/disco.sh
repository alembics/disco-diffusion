#!/bin/bash
# One-liner while using Docker during development
# Meant to be run from the home directory of this repo
# Purge garbage layers
docker image prune -f && \
# Re-build the docker file
docker build -t disco-diffusion:dev --file ../docker/main/Dockerfile . && \
# Run a test docker command, adding whatever is currently in your disco-diffusion cwd.
# Make note to change the volume mappings to reflect your host env
docker run --rm -it \
    --gpus device=all --cpus=2.0 --name="unittest-disco" --ipc=host --user $(id -u):$(id -g) \
    -v /home/mike/ai/disco5/images_out:/workspace/code/images_out \
    -v /home/mike/ai/disco5/init_images:/workspace/code/init_images \
    -v /home/mike/disco-diffusion-1:/workspace/scratch \
    -e cuda_device='"cuda:0"' \
    disco-diffusion:dev bash -c "cp -r /workspace/scratch/* /workspace/code/disco-diffusion-1/ && 
    python disco-diffusion-1/disco.py --cuda_device=cpu --RN50=false --ViTB16=false --ViTB32=false"
