# Docker

## Introduction

This is a Docker build file that will preinstall dependencies, packages, Git repos, and pre-cache the large model files needed by Disco Diffusion.

## TO-DO:

- Make container actually accept parameters on run.  Right now you'll just be seeing lighthouses.

## Change Log

- `1.0`

  Initial build file created based on the DD 5.1 Git repo.  This initial build is deliberately meant to work touch-free of any of the existing Python code written.  It does handle some of the pre-setup tasks already done in the Python code such as pip packages, Git clones, and even pre-caching the model files for faster launch speed.

## Build the Prep Image
The prep image is broken out from the `main` folder's `Dockerfile` to help with long build context times (or wget download times after intitial build.)  This prep image build contains all the large model files required by Disco Diffusion.

From a terminal in the `docker/prep` directory, run:
```sh
docker build -t disco-diffusion-prep:5.1 .
```
From a terminal in the `docker/main` directory, run:
## Build the Image
From a terminal, run:

```sh
docker build -t disco-diffusion:5.1 .
```

## Run as a Container

This example runs Disco Diffusion in a Docker container.  It maps `images_out` and `init_images` to the container's working directory to access by the host OS.
```sh
docker run --rm -it \
    -v $(echo ~)/disco-diffusion/images_out:/workspace/code/images_out \
    -v $(echo ~)/disco-diffusion/init_images:/workspace/code/init_images \
    --gpus=all \
    --name="disco-diffusion" --ipc=host \
    --user $(id -u):$(id -g) \
disco-diffusion:5.1 python disco-diffusion/disco.py
```

## Passing Parameters

This will be added after conferring with repo authors.