import os
import subprocess, os, sys
import pathlib, shutil
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

# Set base project directory to current working directory
PROJECT_DIR = os.path.abspath(os.getcwd())
sys.path.append(f'{PROJECT_DIR}')

# Install any missing Git deps

if not os.path.exists(f'{PROJECT_DIR}/pytorch3d-lite'):
  git_output = subprocess.run('git clone https://github.com/MSFTserver/pytorch3d-lite.git'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')

sys.path.append(f'{PROJECT_DIR}/pytorch3d-lite')

from dd import *

# warnings.filterwarnings("ignore", category=UserWarning)
console_preview=False #@param {type:"boolean"}
console_preview_width=80
simple_nvidia_smi_display = False #@param {type:"boolean"}
cuda_device='cuda:0'
is_colab = False
google_drive = False
save_models_to_google_drive = False
root_path = os.getcwd()
initDirPath = f'{root_path}/init_images'
createPath(initDirPath)
outDirPath = f'{root_path}/images_out'
createPath(outDirPath)
model_path = f'{root_path}/models'
createPath(model_path)
createPath(f'{PROJECT_DIR}/pretrained')

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

USE_ADABINS = True

root_path = os.getcwd()
model_path = f'{root_path}/models'

model_512_downloaded = False
model_secondary_downloaded = False

# Download models if not present
for m in [{'file' :f'{model_path}/dpt_large-midas-2f21e586.pt', 'url':'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt'},
          {'file' :f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt', 'url':'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'},
          {'file' :f'{model_path}/256x256_diffusion_uncond.pt', 'url':'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'},
          {'file' :f'{model_path}/secondary_model_imagenet_2.pth', 'url':'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'},
          {'file' :f'{PROJECT_DIR}/pretrained/AdaBins_nyu.pt', 'url':'https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt'}
]:
  if not os.path.exists(f'{m["file"]}'):
    print(f'🌍 (First time setup): Downloading model from {m["url"]} to {m["file"]}')
    wget.download(m["url"], model_path)

os.chdir(f'{PROJECT_DIR}')

if simple_nvidia_smi_display:
  nvidiasmi_output = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_output)
else:
  nvidiasmi_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_output)
  nvidiasmi_ecc_note = subprocess.run(['nvidia-smi', '-i', '0'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  print(nvidiasmi_ecc_note)
  
DEVICE = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
device = DEVICE # At least one of the modules expects this name..
if torch.cuda.get_device_capability(DEVICE) == (8,0): ## A100 fix thanks to Emad
  print('Disabling CUDNN for A100 gpu', file=sys.stderr)
  torch.backends.cudnn.enabled = False

# stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete

TRANSLATION_SCALE = 1.0/200.0
MAX_ADABINS_AREA = 500000

diffusion_model = "512x512_diffusion_uncond_finetune_008100" #@param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100"]
use_secondary_model = True #@param {type: 'boolean'}
diffusion_sampling_mode = 'ddim' #@param ['plms','ddim']  

cutout_debug = False #@param {type: 'boolean'}
use_checkpoint = True #@param {type: 'boolean'}
ViTB32 = True #@param{type:"boolean"}
ViTB16 = True #@param{type:"boolean"}
ViTL14 = False #@param{type:"boolean"}
ViTL14_336 = False #@param{type:"boolean"}
RN101 = False #@param{type:"boolean"}
RN50 = True #@param{type:"boolean"}
RN50x4 = False #@param{type:"boolean"}
RN50x16 = False #@param{type:"boolean"}
RN50x64 = False #@param{type:"boolean"}
batch_name = 'TimeToDisco' #@param{type: 'string'}
steps = 250 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
width_height = [1280, 768]#@param{type: 'raw'}
clip_guidance_scale = 5000 #@param{type: 'number'}
tv_scale =  0#@param{type: 'number'}
range_scale =   150#@param{type: 'number'}
sat_scale =   0#@param{type: 'number'}
cutn_batches = 4  #@param{type: 'number'}
skip_augs = False#@param{type: 'boolean'}
init_image = None #@param{type: 'string'}
init_scale = 1000 #@param{type: 'integer'}
skip_steps = 10 #@param{type: 'integer'}
animation_mode = 'None' #@param ['None', '2D', '3D', 'Video Input'] {type:'string'}
video_init_path = "training.mp4" #@param {type: 'string'}
extract_nth_frame = 2 #@param {type: 'number'}
video_init_seed_continuity = True #@param {type: 'boolean'}
key_frames = True #@param {type:"boolean"}
max_frames = 10000#@param {type:"number"}
interp_spline = 'Linear' #Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
angle = "0:(0)"#@param {type:"string"}
zoom = "0: (1), 10: (1.05)"#@param {type:"string"}
translation_x = "0: (0)"#@param {type:"string"}
translation_y = "0: (0)"#@param {type:"string"}
translation_z = "0: (10.0)"#@param {type:"string"}
rotation_3d_x = "0: (0)"#@param {type:"string"}
rotation_3d_y = "0: (0)"#@param {type:"string"}
rotation_3d_z = "0: (0)"#@param {type:"string"}
midas_depth_model = "dpt_large"#@param {type:"string"}
midas_weight = 0.3#@param {type:"number"}
near_plane = 200#@param {type:"number"}
far_plane = 10000#@param {type:"number"}
fov = 40#@param {type:"number"}
padding_mode = 'border'#@param {type:"string"}
sampling_mode = 'bicubic'#@param {type:"string"}
turbo_mode = False #@param {type:"boolean"}
turbo_steps = "3" #@param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 10 # frames
frames_scale = 1500 #@param{type: 'integer'}
frames_skip_steps = '60%' #@param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}
vr_mode = False #@param {type:"boolean"}
vr_eye_angle = 0.5 #@param{type:"number"}
vr_ipd = 5.0 #@param{type:"number"}
intermediate_saves = 0#@param{type: 'raw'}
intermediates_in_subfolder = True #@param{type: 'boolean'}
perlin_init = False  #@param{type: 'boolean'}
perlin_mode = 'mixed' #@param ['mixed', 'color', 'gray']
set_seed = 'random_seed' #@param{type: 'string'}
eta = 0.8#@param{type: 'number'}
clamp_grad = True #@param{type: 'boolean'}
clamp_max = 0.05 #@param{type: 'number'}
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05
cut_overview = "[12]*400+[4]*600" #@param {type: 'string'}       
cut_innercut ="[4]*400+[12]*600"#@param {type: 'string'}  
cut_ic_pow = 1#@param {type: 'number'}  
cut_icgray_p = "[0.2]*400+[0]*600"#@param {type: 'string'}
text_prompts = {
    0: ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.", "yellow color scheme"],
    100: ["This set of prompts start at frame 100","This prompt has weight five:5"],
}
image_prompts = { }
display_rate =  50 #@param{type: 'number'}
n_batches =  50 #@param{type: 'number'}
batch_size = 1 
resume_run = False #@param{type: 'boolean'}
run_to_resume = 'latest' #@param{type: 'string'}
resume_from_frame = 'latest' #@param{type: 'string'}
retain_overwritten_frames = False #@param{type: 'boolean'}
skip_video_for_run_all = True #@param {type: 'boolean'}
check_model_SHA = False #@param{type:"boolean"}
#
# Override Notebook defaults if external parameters were provided.
#
for param in ["cuda_device", "simple_nvidia_smi_display", "console_preview", "console_preview_width",
              "diffusion_model", "use_secondary_model", "ViTB32", "ViTB16", "ViTL14", "ViTL14_336",
              "RN101", "RN50", "RN50x4", "RN50x64", "check_model_SHA",
              "batch_name", "steps", "width_height", "clip_guidance_scale", "tv_scale", 
              "range_scale", "sat_scale", "cutn_batches", "skip_augs",
              "init_image", "init_scale", "skip_steps",
              "animation_mode","extract_nth_frame", "video_init_seed_continuity",
              "key_frames", "max_frames","interp_spline", "angle", "zoom",
              "translation_x", "translation_y", "translation_z", "rotation_3d_x", "rotation_3d_y",
              "rotation_3d_z", "midas_depth_model", "midas_weight", "near_plane", "far_plane", "fov",
              "padding_mode", "sampling_mode","turbo_mode", "turbo_steps", "turbo_preroll",
              "frames_scale", "frames_skip_steps","vr_mode", "vr_eye_angle", "vr_ipd",
              "intermediate_saves", "intermediates_in_subfolder",
              "perlin_init", "perlin_mode", "set_seed", "eta", "clamp_grad", "clamp_max",
              "randomize_class", "clip_denoised", "fuzzy_prompt", "rand_mag",
              "cut_overview", "cut_innercut", "cut_ic_pow", "cut_icgray_p",
              "text_prompts", "image_prompts","display_rate", "n_batches",
              "resume_run", "run_to_resume", "resume_from_frame", "retain_overwritten_frames",
              "skip_video_for_run_all","cutout_debug"]:
  globals()[param]=get_param(param,globals()[param])

# Diffuse
# Do the Run!
# 'n_batches' (ignored with animation modes.)


model_config = model_and_diffusion_defaults()
if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
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
        'use_checkpoint': use_checkpoint,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
elif diffusion_model == '256x256_diffusion_uncond':
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
        'use_checkpoint': use_checkpoint,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })

model_default = model_config['image_size']

# Get corrected sizes
side_x = (width_height[0]//64)*64
side_y = (width_height[1]//64)*64
if side_x != width_height[0] or side_y != width_height[1]:
  print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

# Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps
})

videoFramesFolder=None
partialFolder=None

# Make folder for batch
batchFolder = f'{outDirPath}/{batch_name}'
createPath(batchFolder)

if animation_mode == "Video Input":
  videoFramesFolder = f'videoFrames'
  createPath(videoFramesFolder)
  print(f"Exporting Video Frames (1 every {extract_nth_frame})...")
  max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))
  try:
    for f in pathlib.Path(f'{videoFramesFolder}').glob('*.jpg'):
      f.unlink()
  except:
    print('')
  vf = f'select=not(mod(n\,{extract_nth_frame}))'
  subprocess.run(['ffmpeg', '-i', f'{video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{videoFramesFolder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  #!ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg
  
# Insist turbo be used only w 3d anim.
if animation_mode != '3D' and (turbo_mode or vr_mode):
  print('⚠️ Turbo/VR modes only available with 3D animations. Disabling... ⚠️')
  turbo_mode = False
  vr_mode = False

if type(intermediate_saves) is not list:
  if intermediate_saves:
    steps_per_checkpoint = math.floor((steps - skip_steps - 1) // (intermediate_saves+1))
    steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
    print(f'Will save every {steps_per_checkpoint} steps')
  else:
    steps_per_checkpoint = steps+10
else:
  steps_per_checkpoint = None

if intermediate_saves and intermediates_in_subfolder is True:
  partialFolder = f'{batchFolder}/partials'
  createPath(partialFolder)
  
# Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})

if retain_overwritten_frames is True:
  retainFolder = f'{batchFolder}/retained'
  createPath(retainFolder)

if cutout_debug is True:
  cutoutDebugFolder = f'{batchFolder}/debug'
  createPath(cutoutDebugFolder)

skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

if steps <= calc_frames_skip_steps:
  sys.exit("⚠️ ERROR: You can't skip more steps than your total steps ⚠️")

if resume_run:
  if run_to_resume == 'latest':
    try:
      batchNum # type: ignore
    except:
      batchNum = len(glob(f"{batchFolder}/{batch_name}(*)_settings.txt"))-1
  else:
    batchNum = int(run_to_resume)
  if resume_from_frame == 'latest':
    start_frame = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
    if animation_mode != '3D' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
      start_frame = start_frame - (start_frame % int(turbo_steps))
  else:
    start_frame = int(resume_from_frame)+1
    if animation_mode != '3D' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
      start_frame = start_frame - (start_frame % int(turbo_steps))
    if retain_overwritten_frames is True:
      existing_frames = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
      frames_to_save = existing_frames - start_frame
      print(f'Moving {frames_to_save} frames to the Retained folder')
      move_files(start_frame, existing_frames, batchFolder, retainFolder, batch_name=batch_name, batchNum=batchNum)
else:
  start_frame = 0
  batchNum = len(glob(batchFolder+"/*.txt"))
  while os.path.isfile(f"{batchFolder}/{batch_name}({batchNum})_settings.txt") is True or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_settings.txt") is True:
    batchNum += 1

print(f'💻 Starting Run: {batch_name}({batchNum}) at frame {start_frame}')

if set_seed == 'random_seed':
    random.seed()
    seed = random.randint(0, 2**32)
    print(f'🌱 Randomly using seed: {seed}')
else:
    seed = int(set_seed)

try:
  args = {
    'cutout_debug':cutout_debug,
    'ViTB32':ViTB32,
    'ViTB16':ViTB16,
    'ViTL14':ViTL14,
    'ViTL14_336':ViTL14_336,
    'RN50':RN50,
    'RN50x4':RN50x4,
    'RN50x16':RN50x16,
    'RN50x64':RN50x64,
    'RN101':RN101,
    'animation_mode' : animation_mode,
    'batchNum': batchNum,
    'prompts_series':text_prompts,
    'image_prompts_series':image_prompts,
    'seed': seed,
    'display_rate':display_rate,
    'n_batches':n_batches if animation_mode == 'None' else 1,
    'batch_size':batch_size,
    'batch_name': batch_name,
    'steps': steps,
    'diffusion_sampling_mode': diffusion_sampling_mode,
    'width_height': width_height,
    'clip_guidance_scale': clip_guidance_scale,
    'tv_scale': tv_scale,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'cutn_batches': cutn_batches,
    'init_image': init_image,
    'init_scale': init_scale,
    'skip_steps': skip_steps,
    'side_x': side_x,
    'side_y': side_y,
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
    'animation_mode': animation_mode,
    'video_init_path': video_init_path,
    'extract_nth_frame': extract_nth_frame,
    'video_init_seed_continuity': video_init_seed_continuity,
    'key_frames': key_frames,
    'max_frames': max_frames if animation_mode != "None" else 1,
    'interp_spline': interp_spline,
    'start_frame': start_frame,
    'angle': angle,
    'zoom': zoom,
    'translation_x': translation_x,
    'translation_y': translation_y,
    'translation_z': translation_z,
    'rotation_3d_x': rotation_3d_x,
    'rotation_3d_y': rotation_3d_y,
    'rotation_3d_z': rotation_3d_z,
    'midas_depth_model': midas_depth_model,
    'midas_weight': midas_weight,
    'near_plane': near_plane,
    'far_plane': far_plane,
    'fov': fov,
    'padding_mode': padding_mode,
    'sampling_mode': sampling_mode,
    'frames_scale': frames_scale,
    'calc_frames_skip_steps': calc_frames_skip_steps,
    'skip_step_ratio': skip_step_ratio,
    'calc_frames_skip_steps': calc_frames_skip_steps,
    'text_prompts': text_prompts,
    'image_prompts': image_prompts,
    'cut_overview': cut_overview,
    'cut_innercut': cut_innercut,
    'cut_ic_pow': cut_ic_pow,
    'cut_icgray_p': cut_icgray_p,
    'intermediate_saves': intermediate_saves,
    'intermediates_in_subfolder': intermediates_in_subfolder,
    'steps_per_checkpoint': steps_per_checkpoint,
    'perlin_init': perlin_init,
    'perlin_mode': perlin_mode,
    'set_seed': set_seed,
    'eta': eta,
    'clamp_grad': clamp_grad,
    'clamp_max': clamp_max,
    'skip_augs': skip_augs,
    'randomize_class': randomize_class,
    'clip_denoised': clip_denoised,
    'fuzzy_prompt': fuzzy_prompt,
    'rand_mag': rand_mag,
    'turbo_mode': turbo_mode,
    'turbo_preroll':turbo_preroll,
    'turbo_steps':turbo_steps,
    'video_init_seed_continuity':video_init_seed_continuity,
    'videoFramesFolder':videoFramesFolder,
    'TRANSLATION_SCALE':TRANSLATION_SCALE,
    'use_secondary_model':use_secondary_model,
    'diffusion_model':diffusion_model,
    'console_preview':console_preview,
    'console_preview_width':console_preview_width,
    'partialFolder':partialFolder,
    'model_path':model_path,
    'batchFolder':batchFolder,
    'resume_run':resume_run
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

if animation_mode != 'None':
  if skip_video_for_run_all == True:
    print('⚠️ Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
  else:
    createVideo(args)