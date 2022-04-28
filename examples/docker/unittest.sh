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
    -e simple_nvidia_smi_display='true' \
    -e batch_name='"unit-test"' \
    -e width_height='[192,128]' \
    -e steps=100 \
    -e n_batches=2 \
    -e display_rate=50 \
    -e console_preview='true' \
    -e console_preview_width=120 \
    -e set_seed=8675309 \
    -e text_prompts='{"0":["robots testing continuous integration, github, devops and automated testing"]}' \
    disco-diffusion:dev bash -c "cp -r /workspace/scratch/* /workspace/code/disco-diffusion-1/ && python disco-diffusion-1/disco.py"
