# Disco Diffusion

<a href="https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

A frankensteinian amalgamation of notebooks, models and techniques for the generation of AI Art and Animations.

[to be updated with further info soon]

## Contributing
This project uses a special conversion tool to convert the python files into notebooks for easier development.

What this means is you do not have to touch the notebook directly to make changes to it

the tool being used is called [Colab-Convert](https://github.com/MSFTserver/colab-convert)

- install using `pip install colab-convert`
- convert .py to .ipynb `colab-convert /path/to/file.py /path/to/file.ipynb`
- convert .ipynb to .py `colab-convert /path/to/file.ipynb /path/to/file.py`


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

#### v4 Update: Jan 2022 - Somnai
* Implemented Diffusion Zooming
* Added Chigozie keyframing
* Made a bunch of edits to processes

#### v4.1 Update: Jan 14th 2022 - Somnai
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

#### v5.3 Update: Jun 10th 2022 - nshepperd, huemin, cut_pow

* Horizontal and Vertical symmetry
* Addition of ViT-L/14@336px model (requires high VRAM)

#### v5.4 Update: Jun 14th 2022 - devdef / Alex Spirin, integrated into DD main by gandamu / Adam Letts

* Warp mode - for smooth/continuous video input results leveraging optical flow estimation and frame blending
* Custom models support

#### v5.5 Update: Jul 11th 2022 - Palmweaver / Chris Scalf, KaliYuga_ai, further integration by gandamu / Adam Letts

* OpenCLIP models integration
* Pixel Art Diffusion, Watercolor Diffusion, and Pulp SciFi Diffusion models
* cut_ic_pow scheduling

#### v5.6 Update: Jul 13th 2022 - Felipe3DArtist, integration by gandamu / Adam Letts

* Integrated portrait_generator_v001 - 512x512 diffusion model trained on faces - from Felipe3DArtist

## Notebook Provenance

Original notebook by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings). It uses either OpenAI's 256x256 unconditional ImageNet or Katherine Crowson's fine-tuned 512x512 diffusion model (https://github.com/openai/guided-diffusion), together with CLIP (https://github.com/openai/CLIP) to connect text prompts with images.

Modified by Daniel Russell (https://github.com/russelldc, https://twitter.com/danielrussruss) to include (hopefully) optimal params for quick generations in 15-100 timesteps rather than 1000, as well as more robust augmentations.

Further improvements from Dango233 and nshepperd helped improve the quality of diffusion in general, and especially so for shorter runs like this notebook aims to achieve.

Vark added code to load in multiple Clip models at once, which all prompts are evaluated against, which may greatly improve accuracy.

The latest zoom, pan, rotation, and keyframes features were taken from Chigozie Nri's VQGAN Zoom Notebook (https://github.com/chigozienri, https://twitter.com/chigozienri)

Advanced DangoCutn Cutout method is also from Dango223.

--

Somnai (https://twitter.com/Somnai_dreams) added 2D Diffusion animation techniques, QoL improvements and various implementations of tech and techniques, mostly listed in the changelog below.

3D animation implementation added by Adam Letts (https://twitter.com/gandamu_ml) in collaboration with Somnai.

Turbo feature by Chris Allen (https://twitter.com/zippy731)

Improvements to ability to run on local systems, Windows support, and dependency installation by HostsServer (https://twitter.com/HostsServer)

VR Mode by Tom Mason (https://twitter.com/nin_artificial)

Horizontal and Vertical symmetry functionality by nshepperd. Symmetry transformation_steps by huemin (https://twitter.com/huemin_art). Symmetry integration into Disco Diffusion by Dmitrii Tochilkin (https://twitter.com/cut_pow).

Warp and custom model support by Alex Spirin (https://twitter.com/devdef).

Pixel Art Diffusion, Watercolor Diffusion, and Pulp SciFi Diffusion models from KaliYuga (https://twitter.com/KaliYuga_ai). Follow KaliYuga's Twitter for the latest models and for notebooks with specialized settings.

Integration of OpenCLIP models and initiation of integration of KaliYuga models by Palmweaver / Chris Scalf (https://twitter.com/ChrisScalf11)

Integrated portrait_generator_v001 from Felipe3DArtist (https://twitter.com/Felipe3DArtist)