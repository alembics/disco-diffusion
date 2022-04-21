# Disco Diffusion

<a href="https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

A frankensteinian amalgamation of notebooks, models and techniques for the generation of AI Art and Animations.

<img src="images_out/TimeToDisco/TimeToDisco(0)_0.png" />

## Changes in this Fork

- Focus on running from Windows or Linux CLI instead of keeping Notebook parity in the short term
- Move all functions possible out of the main `disco.py` module into `dd.py` so that `disco.py` can become readable.  (Ongoing work in progress here)
- Instead changing parameters directly in `disco.py`, parameters can be controlled by environment variables
- **TODO:** Allow JSON/CSV/YAML parameter files to be used as inputs

## Windows First-time Setup (Anaconda)

Follow these steps for the first time that you are running Disco Diffusion from Windows.

### Pre-requisites

- Anaconda installed
- Nvidia CUDA Toolkit Installed
- MS VS Community Installed with C++ checkmarked

1. From **Anaconda Powershell Prompt**:
    
    This command will allow you to use `conda` from a "regular" powershell session.
    ```
    conda init powershell
    exit
    ```
2. From your **VS Code Powershell prompt**:

    This command will pull all dependencies needed by Disco Diffusion into a conda environment called `discodiffusion`
    ```
    conda env create -f environment.yml
    conda activate discodiffusion
    ```

3. Compile `pytorch3d`

    For reason I'm not 100% clear on, `pytorch3d` must be compiled in Windows.  (Hence the requirement for C++ tool mentioned in Pre-requisties...)
    ```
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    python setup.py install
    cd ..
    ```
4. Execute a test run:

    The following test run will run with all defaults (so "the lighthouse run" as it is coined.)  Image output and current image progress (`progress.png`) will be stored in `images_out`.
    ```
    conda activate discodiffusion
    python disco.py
    ```

## Changelog
#### v1 Oct 29th 2021 - Somnai  
* Initial QoL improvements added, including user friendly UI, settings+prompt saving and improved google drive folder organization.

#### v1.1 Nov 13th 2021 - Somnai
* Now includes sizing options, intermediate saves and fixed image prompts and perlin inits. unexposed batch option since it doesn't work

#### v2 Update: Nov 22nd 2021 - Somnai
* Initial addition of Katherine Crowson's Secondary Model Method (https://colab.research.google.com/drive/1mpkrhOjoyzPeSWy2r7T8EYRaU7amYOOi#scrollTo=X5gODNAMEUCR)
* Fix for incorrectly named settings files

#### v3 Update: Dec 24th 2021 - Somnai
* Implemented Dango's advanced cutout method
* Added SLIP models, thanks to NeuralDivergent
* Fixed issue with NaNs resulting in black images, with massive help and testing from @Softology
* Perlin now changes properly within batches (not sure where this perlin_regen code came from originally, but thank you)

#### v4 Update: Jan 2021 - Somnai
* Implemented Diffusion Zooming
* Added Chigozie keyframing
* Made a bunch of edits to processes

#### v4.1 Update: Jan 14th 2021 - Somnai
* Added video input mode
* Added license that somehow went missing
* Added improved prompt keyframing, fixed image_prompts and multiple prompts
* Improved UI
* Significant under the hood cleanup and improvement
* Refined defaults for each mode
* Removed SLIP models for the time being due to import conflicts
* Added latent-diffusion SuperRes for sharpening
* Added resume run mode

#### v5 Update: Feb 20th 2022 - gandamu / Adam Letts
* Added 3D animation mode. Uses weighted combination of AdaBins and MiDaS depth estimation models. Uses pytorch3d for 3D transforms on Colab and/or Linux.

#### v5.1 Update: Mar 30th 2022 - zippy / Chris Allen and gandamu / Adam Letts

* Integrated Turbo+Smooth features from Disco Diffusion Turbo -- just the implementation, without its defaults.
* Implemented resume of turbo animations in such a way that it's now possible to resume from different batch folders and batch numbers.
* 3D rotation parameter units are now degrees (rather than radians)
* Corrected name collision in sampling_mode (now diffusion_sampling_mode for plms/ddim, and sampling_mode for 3D transform sampling)
* Added video_init_seed_continuity option to make init video animations more continuous
* Removed pytorch3d from needing to be compiled with a lite version specifically made for Disco Diffusion
* Remove Super Resolution
* Remove Slip Models
* Update for crossplatform support

#### v5.1 Update: Apr 4th 2022 - MSFTserver aka HostsServer

* Removed pytorch3d from needing to be compiled with a lite version specifically made for Disco Diffusion
* Remove Super Resolution
* Remove Slip Models
* Update for crossplatform support

#### v5.2 Update: Apr 10th 2022 - nin_artificial / Tom Mason

* VR Mode

## Notebook Provenance 

Original notebook by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings). It uses either OpenAI's 256x256 unconditional ImageNet or Katherine Crowson's fine-tuned 512x512 diffusion model (https://github.com/openai/guided-diffusion), together with CLIP (https://github.com/openai/CLIP) to connect text prompts with images.

Modified by Daniel Russell (https://github.com/russelldc, https://twitter.com/danielrussruss) to include (hopefully) optimal params for quick generations in 15-100 timesteps rather than 1000, as well as more robust augmentations.

Further improvements from Dango233 and nsheppard helped improve the quality of diffusion in general, and especially so for shorter runs like this notebook aims to achieve.

Vark added code to load in multiple Clip models at once, which all prompts are evaluated against, which may greatly improve accuracy.

The latest zoom, pan, rotation, and keyframes features were taken from Chigozie Nri's VQGAN Zoom Notebook (https://github.com/chigozienri, https://twitter.com/chigozienri)

Advanced DangoCutn Cutout method is also from Dango223.

--

Somnai (https://twitter.com/Somnai_dreams) added 2D Diffusion animation techniques, QoL improvements and various implementations of tech and techniques, mostly listed in the changelog below.

3D animation implementation added by Adam Letts (https://twitter.com/gandamu_ml) in collaboration with Somnai.
