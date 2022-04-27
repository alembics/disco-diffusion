docker run --rm -it \
    --gpus device=GPU-32e74b94-5a45-b62c-68ba-3686ab970333 --name="disco-diffusion1" --ipc=host --user $(id -u):$(id -g) \
    -v /home/mike/ai/disco5/images_out2:/workspace/code/images_out \
    -v /home/mike/ai/disco5/init_images:/workspace/code/init_images \
    -v /home/mike/code/other/disco-diffusion:/workspace/scratch \
    -e batch_name='"fantasy"' \
    -e xanimation_mode='"2D"' \
    -e width_height='[1280,768]' \
    -e steps=500 \
    -e display_rate=10 \
    -e set_seed=509896214 \
    -e RN50x64=true \
    -e RN50=false \
    -e text_prompts='{"0":["fairy tale castles in a fantasy forest, rainbows over meadows of emerald grass, toadstools and pots of gold, unreal engine, trending on artstation, art by lisa frank"]}' \
    -e init_scale=1 \
    disco-diffusion:5.1 bash -c \
    "cp /workspace/scratch/disco.py disco-diffusion/ && cp /workspace/scratch/dd.py disco-diffusion/ && python disco-diffusion/disco.py"