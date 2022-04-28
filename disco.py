import os
import subprocess, os, sys
import pathlib, shutil
from numpy import True_
import torch
import gc
import math
from glob import glob
import random
import torch
import warnings
import wget
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from pydotted import pydot

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Set base project directory to current working directory
PROJECT_DIR = os.path.abspath(os.getcwd())
USE_ADABINS = True
TRANSLATION_SCALE = 1.0/200.0
MAX_ADABINS_AREA = 500000

sys.path.append(f'{PROJECT_DIR}')

is_colab = False
google_drive = False
save_models_to_google_drive = False

# Install pytorch3d-lite
if not os.path.exists(f'{PROJECT_DIR}/pytorch3d-lite'):
  git_output = subprocess.run('git clone https://github.com/MSFTserver/pytorch3d-lite.git'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
sys.path.append(f'{PROJECT_DIR}/pytorch3d-lite')

from dd import *
import dd_args

pargs = dd_args.parse()

# warnings.filterwarnings("ignore", category=UserWarning)

videoFramesFolder=None
partialFolder=None
root_path = os.getcwd()
initDirPath = f'{root_path}/init_images'
outDirPath = f'{root_path}/images_out'
model_path = f'{root_path}/models'
batchFolder = f'{outDirPath}/{pargs.batch_name}'

createPath(initDirPath)
createPath(outDirPath)
createPath(model_path)
createPath(f'{PROJECT_DIR}/pretrained')
createPath(batchFolder)


# Download models if not present
for m in [{'file' :f'{model_path}/dpt_large-midas-2f21e586.pt', 'url':'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt'},
          {'file' :f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt', 'url':'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'},
          {'file' :f'{model_path}/256x256_diffusion_uncond.pt', 'url':'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'},
          {'file' :f'{model_path}/secondary_model_imagenet_2.pth', 'url':'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'},
          {'file' :f'{PROJECT_DIR}/pretrained/AdaBins_nyu.pt', 'url':'https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt'}
]:
  if not os.path.exists(f'{m["file"]}'):
    print(f'🌍 (First time setup): Downloading model from {m["url"]} to {m["file"]}')
    wget.download(m["url"], m["file"])

if pargs.simple_nvidia_smi_display:
  nvidiasmi_output = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_output)
else:
  nvidiasmi_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_output)
  nvidiasmi_ecc_note = subprocess.run(['nvidia-smi', '-i', '0'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_ecc_note)
  
DEVICE = torch.device(pargs.cuda_device if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
device = DEVICE # At least one of the modules expects this name..
if torch.cuda.get_device_capability(DEVICE) == (8,0): ## A100 fix thanks to Emad
  print('Disabling CUDNN for A100 gpu', file=sys.stderr)
  torch.backends.cudnn.enabled = False

model_config = model_and_diffusion_defaults()
if pargs.diffusion_model == '512x512_diffusion_uncond_finetune_008100':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': pargs.use_checkpoint,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
elif pargs.diffusion_model == '256x256_diffusion_uncond':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': pargs.use_checkpoint,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })

model_default = model_config['image_size']

# Get corrected sizes
side_x = (pargs.width_height[0]//64)*64
side_y = (pargs.width_height[1]//64)*64
if side_x != pargs.width_height[0] or side_y != pargs.width_height[1]:
  print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

# Update Model Settings
timestep_respacing = f'ddim{pargs.steps}'
diffusion_steps = (1000//pargs.steps)*pargs.steps if pargs.steps < 1000 else pargs.steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps
})

if pargs.animation_mode == "Video Input":
  videoFramesFolder = f'videoFrames'
  createPath(videoFramesFolder)
  print(f"Exporting Video Frames (1 every {pargs.extract_nth_frame})...")
  pargs.max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))
  try:
    for f in pathlib.Path(f'{videoFramesFolder}').glob('*.jpg'):
      f.unlink()
  except:
    print('')
  vf = f'select=not(mod(n\,{pargs.extract_nth_frame}))'
  subprocess.run(['ffmpeg', '-i', f'{pargs.video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{videoFramesFolder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  #!ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg
  
# Insist turbo be used only w 3d anim.
if pargs.animation_mode != '3D' and (pargs.turbo_mode or pargs.vr_mode):
  print('⚠️ Turbo/VR modes only available with 3D animations. Disabling... ⚠️')
  pargs.turbo_mode = False
  pargs.vr_mode = False

if type(pargs.intermediate_saves) is not list:
  if pargs.intermediate_saves:
    steps_per_checkpoint = math.floor((pargs.steps - pargs.skip_steps - 1) // (pargs.intermediate_saves+1))
    steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
    print(f'Will save every {steps_per_checkpoint} steps')
  else:
    steps_per_checkpoint = pargs.steps+10
else:
  steps_per_checkpoint = None

if pargs.intermediate_saves and pargs.intermediates_in_subfolder is True:
  partialFolder = f'{batchFolder}/partials'
  createPath(partialFolder)
  
# Update Model Settings
timestep_respacing = f'ddim{pargs.steps}'
diffusion_steps = (1000//pargs.steps)*pargs.steps if pargs.steps < 1000 else pargs.steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})

if pargs.retain_overwritten_frames is True:
  retainFolder = f'{batchFolder}/retained'
  createPath(retainFolder)

if pargs.cutout_debug is True:
  cutoutDebugFolder = f'{batchFolder}/debug'
  createPath(cutoutDebugFolder)

skip_step_ratio = int(pargs.frames_skip_steps.rstrip("%")) / 100
calc_frames_skip_steps = math.floor(pargs.steps * skip_step_ratio)

if pargs.steps <= calc_frames_skip_steps:
  sys.exit("⚠️ ERROR: You can't skip more steps than your total steps ⚠️")

if pargs.resume_run:
  if pargs.run_to_resume == 'latest':
    try:
      batchNum # type: ignore
    except:
      batchNum = len(glob(f"{batchFolder}/{pargs.batch_name}(*)_settings.txt"))-1
  else:
    batchNum = int(pargs.run_to_resume)
  if pargs.resume_from_frame == 'latest':
    start_frame = len(glob(batchFolder+f"/{pargs.batch_name}({batchNum})_*.png"))
    if pargs.animation_mode != '3D' and pargs.turbo_mode == True and start_frame > pargs.turbo_preroll and start_frame % int(pargs.turbo_steps) != 0:
      start_frame = start_frame - (start_frame % int(pargs.turbo_steps))
  else:
    start_frame = int(pargs.resume_from_frame)+1
    if pargs.animation_mode != '3D' and pargs.turbo_mode == True and start_frame > pargs.urbo_preroll and start_frame % int(pargs.turbo_steps) != 0:
      start_frame = start_frame - (start_frame % int(pargs.turbo_steps))
    if pargs.retain_overwritten_frames is True:
      existing_frames = len(glob(batchFolder+f"/{pargs.batch_name}({batchNum})_*.png"))
      frames_to_save = existing_frames - start_frame
      print(f'Moving {frames_to_save} frames to the Retained folder')
      move_files(start_frame, existing_frames, batchFolder, retainFolder, batch_name=pargs.batch_name, batchNum=batchNum)
else:
  start_frame = 0
  batchNum = len(glob(batchFolder+"/*.txt"))
  while os.path.isfile(f"{batchFolder}/{pargs.batch_name}({batchNum})_settings.txt") is True or os.path.isfile(f"{batchFolder}/{pargs.batch_name}-{batchNum}_settings.txt") is True:
    batchNum += 1

print(f'💻 Starting Run: {pargs.batch_name}({batchNum}) at frame {start_frame}')

if pargs.set_seed == 'random_seed':
    random.seed()
    seed = random.randint(0, 2**32)
    print(f'🌱 Randomly using seed: {seed}')
else:
    seed = int(pargs.set_seed)

try:
  args = {
    'use_checkpoint': pargs.use_checkpoint,
    'cutout_debug': pargs.cutout_debug,
    'ViTB32': pargs.ViTB32,
    'ViTB16': pargs.ViTB16,
    'ViTL14': pargs.ViTL14,
    'ViTL14_336': pargs.ViTL14_336,
    'RN50': pargs.RN50,
    'RN50x4': pargs.RN50x4,
    'RN50x16': pargs.RN50x16,
    'RN50x64': pargs.RN50x64,
    'RN101': pargs.RN101,
    'diffusion_sampling_mode': pargs.diffusion_sampling_mode,
    'width_height': pargs.width_height,
    'clip_guidance_scale': pargs.clip_guidance_scale,
    'tv_scale': pargs.tv_scale,
    'range_scale': pargs.range_scale,
    'sat_scale': pargs.sat_scale,
    'cutn_batches': pargs.cutn_batches,
    'use_secondary_model': pargs.use_secondary_model,
    'diffusion_model': pargs.diffusion_model,
    'animation_mode' : pargs.animation_mode,
    'batchNum': batchNum,
    'prompts_series': pargs.text_prompts,
    'text_prompts': pargs.text_prompts,
    'console_preview':pargs.console_preview,
    'console_preview_width':pargs.console_preview_width,
    'image_prompts_series': pargs.image_prompts,
    'seed': seed,
    'display_rate': pargs.display_rate,
    'n_batches': pargs.n_batches if pargs.animation_mode == 'None' else 1,
    'batch_size': 1,
    'batch_name': pargs.batch_name,
    'steps': pargs.steps,
    'init_image': pargs.init_image,
    'init_scale': pargs.init_scale,
    'skip_steps': pargs.skip_steps,
    'side_x': side_x,
    'side_y': side_y,
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
    'animation_mode': pargs.animation_mode,
    'video_init_path': pargs.video_init_path,
    'extract_nth_frame': pargs.extract_nth_frame,
    'video_init_seed_continuity': pargs.video_init_seed_continuity,
    'key_frames': pargs.key_frames,
    'max_frames': pargs.max_frames if pargs.animation_mode != "None" else 1,
    'interp_spline': pargs.interp_spline,
    'start_frame': start_frame,
    'angle': pargs.angle,
    'zoom': pargs.zoom,
    'translation_x': pargs.translation_x,
    'translation_y': pargs.translation_y,
    'translation_z': pargs.translation_z,
    'rotation_3d_x': pargs.rotation_3d_x,
    'rotation_3d_y': pargs.rotation_3d_y,
    'rotation_3d_z': pargs.rotation_3d_z,
    'midas_depth_model': pargs.midas_depth_model,
    'midas_weight': pargs.midas_weight,
    'near_plane': pargs.near_plane,
    'far_plane': pargs.far_plane,
    'fov': pargs.fov,
    'padding_mode': pargs.padding_mode,
    'sampling_mode': pargs.sampling_mode,
    'frames_scale': pargs.frames_scale,
    'calc_frames_skip_steps': calc_frames_skip_steps,
    'skip_step_ratio': skip_step_ratio,
    'calc_frames_skip_steps': calc_frames_skip_steps,
    'image_prompts': pargs.image_prompts,
    'cut_overview': pargs.cut_overview,
    'cut_innercut': pargs.cut_innercut,
    'cut_ic_pow': pargs.cut_ic_pow,
    'cut_icgray_p': pargs.cut_icgray_p,
    'intermediate_saves': pargs.intermediate_saves,
    'intermediates_in_subfolder': pargs.intermediates_in_subfolder,
    'steps_per_checkpoint': steps_per_checkpoint,
    'perlin_init': pargs.perlin_init,
    'perlin_mode': pargs.perlin_mode,
    'set_seed': pargs.set_seed,
    'eta': pargs.eta,
    'clamp_grad': pargs.clamp_grad,
    'clamp_max': pargs.clamp_max,
    'skip_augs': pargs.skip_augs,
    'randomize_class': pargs.randomize_class,
    'clip_denoised': pargs.clip_denoised,
    'fuzzy_prompt': pargs.fuzzy_prompt,
    'rand_mag': pargs.rand_mag,
    'turbo_mode': pargs.turbo_mode,
    'turbo_preroll': pargs.turbo_preroll,
    'turbo_steps': pargs.turbo_steps,
    'video_init_seed_continuity': pargs.video_init_seed_continuity,
    'videoFramesFolder': videoFramesFolder,
    'TRANSLATION_SCALE': TRANSLATION_SCALE,
    'partialFolder': partialFolder,
    'model_path': model_path,
    'batchFolder': batchFolder,
    'resume_run': pargs.resume_run
  }
  # args = SimpleNamespace(**args)
  args = pydot(args)
  # print(args)
  do_run(args, 
    device=device,
    is_colab=is_colab,
    model_config=model_config)
except KeyboardInterrupt:
  print('🛑 Run interrupted by user.')
  pass
finally:
  gc.collect()
  torch.cuda.empty_cache()

if pargs.animation_mode != 'None':
  if pargs.skip_video_for_run_all == True:
    print('⚠️ Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
  else:
    createVideo(args)