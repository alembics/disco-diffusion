docker run --rm -it \
    --gpus device=GPU-8d085d58-e2f9-35dc-196e-ffa549d84340 --name="disco-diffusion3" --ipc=host --user $(id -u):$(id -g) \
    --cpus="2.0" \
    -v /home/mike/ai/disco5/images_out2:/workspace/code/images_out \
    -v /home/mike/ai/disco5/init_images:/workspace/code/init_images \
    -v /home/mike/code/other/disco-diffusion:/workspace/scratch \
    -e xinit_image='"init_images/canaetto entrahce to the grand canal 4X.jpg"' \
    -e batch_name='"cyberdecay"' \
    -e resume_run='true' \
    -e animation_mode='"2D"' \
    -e xskip_steps=125 \
    -e width_height='[1280,768]' \
    -e steps=500 \
    -e display_rate=10 \
    -e set_seed=509896214 \
    -e tv_scale=10000 \
    -e cut_ic_pow=1 \
    -e xcutn_batches=1 \
    -e xRN50x64=true \
    -e xRN50=false \
    -e text_prompts='{"0":["cybernetic organism, artstation, Art by Beksinski, unreal engine"]}' \
    -e init_scale=1 \
    -e console_preview='true' \
    -e console_preview_width=40 \
    disco-diffusion:dev bash -c \
    "cp -r /workspace/scratch/* disco-diffusion-1/ && python disco-diffusion-1/disco.py"