FROM nvcr.io/nvidia/pytorch:21.08-py3 AS prep
    RUN mkdir -p /scratch/models && \
        mkdir -p /scratch/models/superres && \
        mkdir -p /scratch/models/slip && \
        mkdir -p /scratch/model-lpips && \
        mkdir -p /scratch/clip && \
        mkdir -p /scratch/pretrained

    RUN wget --progress=bar:force:noscroll -P /scratch/model-lpips https://download.pytorch.org/models/vgg16-397923af.pth 

    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/models https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/models https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/models https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/models https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth

    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/pretrained https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt

    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/clip/ https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/clip https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/clip https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/clip https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/clip https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/clip https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/clip https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
    RUN wget --no-directories --progress=bar:force:noscroll -P /scratch/clip https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
