# One-liner while using Docker during development
# Meant to be run from the home directory of this repo
# Purge garbage layers
docker image prune -f && \
# Re-build the docker file
docker build -t disco-diffusion:dev --file ../docker/main/Dockerfile . && \
# Run a test docker command, adding whatever is currently in your disco-diffusion cwd.
# Make note to change the volume mappings to reflect your host env
docker run --rm -it \
    --gpus=all \
    --name="disco-diffusion" --ipc=host \
    -v /home/mike/ai/disco5/images_out:/workspace/code/images_out \
    -v /home/mike/ai/disco5/init_images:/workspace/code/init_images \
    -v /home/mike/code/other/disco-diffusion:/workspace/scratch \
    --user $(id -u):$(id -g) \
    -e batch_name='"sea creatures"' \
    -e text_prompts='{ 
    "0": [
        "artist 1:100", 
        "featured on artstation:50", 
        "science fiction:10",  
        "dystopian:20" 
    ],
    "300": [ 
        "artist 2:100", 
        "featured on artstation:50", 
        "science fiction:10",  
        "dystopian:20" 
    ],
    "600": [ 
        "artist 3:100", 
        "featured on artstation:50", 
        "science fiction:10",  
        "dystopian:20" 
    ],
    "90": [ 
        "artist 4:100", 
        "featured on artstation:50", 
        "science fiction:10",  
        "dystopian:20" 
    ]
    }' \
    -e clip_guidance_scale=15000 \
    -e tv_scale=0  \
    -e range_scale=1500 \
    -e sat_scale=0 \
    -e cutn_batches=4 \
    -e max_frames=10000 \
    -e interp_spline='"Linear"' \
    -e cutn_batches=4 \
    -e clip_guidance_scale=50000 \
    -e cut_ic_pow=1 \
    -e animation_mode='"2D"' \
    -e init_scale=500 \
    -e skip_steps=100 \
    -e frames_scale=1500 \
    -e frames_skip_steps='"60%"' \
    -e use_secondary_model='true' \
    -e steps=1250 \
    -e angle='"0:(0)"' \
    -e zoom='"0: (1), 10: (1.05)"' \
    -e translation_x='"0: (0)"' \
    -e translation_y='"0: (0)"' \
    -e translation_z='"0: (10.0)"' \
    -e rotation_3d_x='"0: (0)"' \
    -e rotation_3d_y='"0: (0)"' \
    -e rotation_3d_z='"0: (0)"' \
    -e midas_depth_model='"dpt_large"' \
    -e midas_weight=0.3 \
    -e near_plane=200 \
    -e far_plane=10000 \
    -e fov=40 \
    -e turbo_mode='false' \
    -e display_rate=20 \
    -e width_height='[1280,768]' \
    -e init_image='"init_images/beale2.jpg"' \
    disco-diffusion:dev bash -c "cp -r /workspace/scratch/* /workspace/code/disco-diffusion-1/ && python disco-diffusion-1/disco.py"
