docker run --rm -it \
    --gpus device=GPU-8d085d58-e2f9-35dc-196e-ffa549d84340 --name="disco-diffusion2" --ipc=host --user $(id -u):$(id -g) \
    --cpus="2.0" \
    -v /home/mike/ai/disco5/images_out2:/workspace/code/images_out \
    -v /home/mike/ai/disco5/init_images:/workspace/code/init_images \
    -v /home/mike/code/other/disco-diffusion:/workspace/scratch \
    -e batch_name='"easter2"' \
    -e resume_run='true' \
    -e animation_mode='"2D"' \
    -e width_height='[1280,768]' \
    -e steps=500 \
    -e display_rate=10 \
    -e set_seed=509896214 \
    -e tv_scale=10000 \
    -e cut_ic_pow=1 \
    -e text_prompts='{"0":["easter eggs, easter rabbits, rainbows and majestic clouds, christ, christianity, trending on artstation"]}' \
    -e init_scale=1 \
    disco-diffusion:5.1 bash -c \
    "cp /workspace/scratch/disco.py disco-diffusion/ && cp /workspace/scratch/dd.py disco-diffusion/ && python disco-diffusion/disco.py"