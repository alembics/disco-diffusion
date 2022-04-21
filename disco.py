import os
import subprocess, os, sys
import subprocess
import pathlib, shutil
import torch
import cv2
import gc
import math
from IPython import display
import lpips
from PIL import Image
from glob import glob
from types import SimpleNamespace
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import torch
import warnings
import climage
import wget

# Set base project directory to current working directory
PROJECT_DIR = os.path.abspath(os.getcwd())
sys.path.append(f'{PROJECT_DIR}')

# Install any missing Git deps
if not os.path.exists(f'{PROJECT_DIR}/CLIP'):
  git_output = subprocess.run('git clone https://github.com/openai/CLIP'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')

if not os.path.exists(f'{PROJECT_DIR}/MiDaS'):
  git_output = subprocess.run('git clone https://github.com/isl-org/MiDaS.git'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')

if not os.path.exists(f'{PROJECT_DIR}/ResizeRight'):
  git_output = subprocess.run('git clone https://github.com/assafshocher/ResizeRight.git'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')

if not os.path.exists(f'{PROJECT_DIR}/pytorch3d-lite'):
  git_output = subprocess.run('git clone https://github.com/MSFTserver/pytorch3d-lite.git'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')

if not os.path.exists(f'{PROJECT_DIR}/AdaBins'):
  git_output = subprocess.run('git clone https://github.com/shariqfarooq123/AdaBins.git'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')

if not os.path.exists(f'{PROJECT_DIR}/guided-diffusion'):
  git_output = subprocess.run('git clone https://github.com/crowsonkb/guided-diffusion.git'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')


# WTF tho
if not os.path.exists('MiDaS/midas_utils.py'):
      shutil.copy('MiDaS/utils.py', 'MiDaS/midas_utils.py')

sys.path.append(f'{PROJECT_DIR}/CLIP')
sys.path.append(f'{PROJECT_DIR}/MiDaS')
sys.path.append(f'{PROJECT_DIR}/ResizeRight')
sys.path.append(f'{PROJECT_DIR}/pytorch3d-lite')
sys.path.append(f'{PROJECT_DIR}/guided-diffusion')
sys.path.append(f'{PROJECT_DIR}/AdaBins')

from dd import *

warnings.filterwarnings("ignore", category=UserWarning)
console_preview=False #@param {type:"boolean"}
console_preview_width=80
simple_nvidia_smi_display = False #@param {type:"boolean"}
cuda_device='cuda:0'
# Override Notebook defaults if external parameters were provided.
for param in ["cuda_device", "simple_nvidia_smi_display", "console_preview", "console_preview_width"]:
  globals()[param]=get_param(param,globals()[param])

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

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

USE_ADABINS = True

root_path = os.getcwd()
model_path = f'{root_path}/models'

model_512_downloaded = False
model_secondary_downloaded = False

# Download models
if not os.path.exists(f'{model_path}/dpt_large-midas-2f21e586.pt'):
  url = "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt"
  print(f'(First time setup): Downloading model from {url}...')
  wget.download(url, model_path)

if not os.path.exists(f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt'):
  url = "https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt"
  print(f'(First time setup): Downloading model from {url}...')
  wget.download(url, model_path)

if not os.path.exists(f'{model_path}/256x256_diffusion_uncond.pt'):
  url = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
  print(f'(First time setup): Downloading model from {url}...')
  wget.download(url, model_path)

if not os.path.exists(f'{model_path}/secondary_model_imagenet_2.pth'):
  url = "https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth"
  print(f'(First time setup): Downloading model from {url}...')
  wget.download(url, model_path)

# WTF tho
# shutil.move('disco-diffusion-1/disco_xform_utils.py', 'disco_xform_utils.py')

from CLIP import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

os.chdir(f'{PROJECT_DIR}')

# AdaBins stuff
if USE_ADABINS:
  try:
    from infer import InferenceHelper
  except:
    if os.path.exists("AdaBins") is not True:
      gitclone("https://github.com/shariqfarooq123/AdaBins.git")
    if not os.path.exists(f'{PROJECT_DIR}/pretrained/AdaBins_nyu.pt'):
      createPath(f'{PROJECT_DIR}/pretrained')
      wget("https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt", f'{PROJECT_DIR}/pretrained')
    sys.path.append(f'{PROJECT_DIR}/AdaBins')
  from infer import InferenceHelper
  MAX_ADABINS_AREA = 500000

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

# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0

def do_run():
  seed = args.seed
  angle = None
  zoom = None
  translation_x = None
  translation_y = None
  # print(range(args.start_frame, args.max_frames))

  if (args.animation_mode == "3D") and (args.midas_weight > 0.0):
      midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model, model_path=model_path, device=device)
  for frame_num in range(args.start_frame, args.max_frames):
      if stop_on_next_loop:
        break
      
      display.clear_output(wait=True)

      # Print Frame progress if animation mode is on
      if args.animation_mode != "None":
        batchBar = tqdm(range(args.max_frames), desc ="Frames")
        batchBar.n = frame_num
        batchBar.refresh()

      
      # Inits if not video frames
      if args.animation_mode != "Video Input":
        if args.init_image == '':
          init_image = None
        else:
          init_image = args.init_image
        init_scale = args.init_scale
        skip_steps = args.skip_steps

      if args.animation_mode == "2D":
        if args.key_frames:
          angle = args.angle_series[frame_num]
          zoom = args.zoom_series[frame_num]
          translation_x = args.translation_x_series[frame_num]
          translation_y = args.translation_y_series[frame_num]
          print(
              f'angle: {angle}',
              f'zoom: {zoom}',
              f'translation_x: {translation_x}',
              f'translation_y: {translation_y}',
          )
        
        if frame_num > 0:
          seed += 1
          if resume_run and frame_num == start_frame:
            img_0 = cv2.imread(batchFolder+f"/{batch_name}({batchNum})_{start_frame-1:04}.png")
          else:
            img_0 = cv2.imread('prevFrame.png')
          center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
          trans_mat = np.float32(
              [[1, 0, translation_x],
              [0, 1, translation_y]]
          )
          rot_mat = cv2.getRotationMatrix2D( center, angle, zoom )
          trans_mat = np.vstack([trans_mat, [0,0,1]])
          rot_mat = np.vstack([rot_mat, [0,0,1]])
          transformation_matrix = np.matmul(rot_mat, trans_mat)
          img_0 = cv2.warpPerspective(
              img_0,
              transformation_matrix,
              (img_0.shape[1], img_0.shape[0]),
              borderMode=cv2.BORDER_WRAP
          )

          cv2.imwrite('prevFrameScaled.png', img_0)
          init_image = 'prevFrameScaled.png'
          init_scale = args.frames_scale
          skip_steps = args.calc_frames_skip_steps

      if args.animation_mode == "3D":
        if frame_num > 0:
          seed += 1    
          if resume_run and frame_num == start_frame:
            img_filepath = batchFolder+f"/{batch_name}({batchNum})_{start_frame-1:04}.png"
            if turbo_mode and frame_num > turbo_preroll:
              shutil.copyfile(img_filepath, 'oldFrameScaled.png')
          else:
            img_filepath = '/content/prevFrame.png' if is_colab else 'prevFrame.png'

          next_step_pil = do_3d_step(img_filepath, frame_num, midas_model, midas_transform, args=args, device=device, TRANSLATION_SCALE=TRANSLATION_SCALE)
          next_step_pil.save('prevFrameScaled.png')

          ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
          if turbo_mode:
            if frame_num == turbo_preroll: #start tracking oldframe
              next_step_pil.save('oldFrameScaled.png')#stash for later blending          
            elif frame_num > turbo_preroll:
              #set up 2 warped image sequences, old & new, to blend toward new diff image
              old_frame = do_3d_step('oldFrameScaled.png', frame_num, midas_model, midas_transform)
              old_frame.save('oldFrameScaled.png')
              if frame_num % int(turbo_steps) != 0: 
                print('turbo skip this frame: skipping clip diffusion steps')
                filename = f'{args.batch_name}({args.batchNum})_{frame_num:04}.png'
                blend_factor = ((frame_num % int(turbo_steps))+1)/int(turbo_steps)
                print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                newWarpedImg = cv2.imread('prevFrameScaled.png')#this is already updated..
                oldWarpedImg = cv2.imread('oldFrameScaled.png')
                blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg,1-blend_factor, 0.0)
                cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                next_step_pil.save(f'{img_filepath}') # save it also as prev_frame to feed next iteration
                continue
              else:
                #if not a skip frame, will run diffusion and need to blend.
                oldWarpedImg = cv2.imread('prevFrameScaled.png')
                cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later 
                print('clip/diff this frame - generate clip diff image')

          init_image = 'prevFrameScaled.png'
          init_scale = args.frames_scale
          skip_steps = args.calc_frames_skip_steps

      if  args.animation_mode == "Video Input":
        if not video_init_seed_continuity:
          seed += 1
        init_image = f'{videoFramesFolder}/{frame_num+1:04}.jpg'
        init_scale = args.frames_scale
        skip_steps = args.calc_frames_skip_steps

      loss_values = []
  
      if seed is not None:
          np.random.seed(seed)
          random.seed(seed)
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = True
  
      target_embeds, weights = [], []
      
      if args.prompts_series is not None and frame_num >= len(args.prompts_series):
        frame_prompt = args.prompts_series[-1]
      elif args.prompts_series is not None:
        frame_prompt = args.prompts_series[frame_num]
      else:
        frame_prompt = []
      
      print(args.image_prompts_series)
      if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
        image_prompt = args.image_prompts_series[-1]
      elif args.image_prompts_series is not None:
        image_prompt = args.image_prompts_series[frame_num]
      else:
        image_prompt = []

      print(f'Frame {frame_num} Prompt: {frame_prompt}')

      model_stats = []
      for clip_model in clip_models:
            cutn = 16
            model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
            model_stat["clip_model"] = clip_model
            
            
            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
                
                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0,1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)
        
            if image_prompt:
              model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs) 
              for prompt in image_prompt:
                  path, weight = parse_prompt(prompt)
                  img = Image.open(fetch(path)).convert('RGB')
                  img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
                  batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                  embed = clip_model.encode_image(normalize(batch)).float()
                  if fuzzy_prompt:
                      for i in range(25):
                          model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0,1))
                          weights.extend([weight / cutn] * cutn)
                  else:
                      model_stat["target_embeds"].append(embed)
                      model_stat["weights"].extend([weight / cutn] * cutn)
        
            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)
  
      init = None
      if init_image is not None:
          init = Image.open(fetch(init_image)).convert('RGB')
          init = init.resize((args.side_x, args.side_y), Image.Resampling.LANCZOS)
          init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
      
      if args.perlin_init:
          if args.perlin_mode == 'color':
              init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False, side_x=side_x, side_y=side_y)
              init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False, side_x=side_x, side_y=side_y)
          elif args.perlin_mode == 'gray':
            init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True, side_x=side_x, side_y=side_y)
            init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True, side_x=side_x, side_y=side_y)
          else:
            init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False, side_x=side_x, side_y=side_y)
            init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True, side_x=side_x, side_y=side_y)
          # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
          init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
          del init2
  
      cur_t = None
  
      def cond_fn(x, t, y=None):
          with torch.enable_grad():
              x_is_NaN = False
              x = x.detach().requires_grad_()
              n = x.shape[0]
              if use_secondary_model is True:
                alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_model(x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
              else:
                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
              for model_stat in model_stats:
                for i in range(args.cutn_batches):
                    t_int = int(t.item())+1 #errors on last step without +1, need to find source
                    #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                    try:
                        input_resolution=model_stat["clip_model"].visual.input_resolution
                    except:
                        input_resolution=224

                    cuts = MakeCutoutsDango(input_resolution,
                      args = args,
                      Overview= args.cut_overview[1000-t_int], 
                      InnerCrop = args.cut_innercut[1000-t_int],
                      IC_Size_Pow=args.cut_ic_pow,
                      IC_Grey_P = args.cut_icgray_p[1000-t_int]
                    )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                    dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                    dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
              tv_losses = tv_loss(x_in)
              if use_secondary_model is True:
                range_losses = range_loss(out)
              else:
                range_losses = range_loss(out['pred_xstart'])
              sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
              loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
              if init is not None and args.init_scale:
                  init_losses = lpips_model(x_in, init)
                  loss = loss + init_losses.sum() * args.init_scale
              x_in_grad += torch.autograd.grad(loss, x_in)[0]
              if torch.isnan(x_in_grad).any()==False:
                  grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
              else:
                # print("NaN'd")
                x_is_NaN = True
                grad = torch.zeros_like(x)
          if args.clamp_grad and x_is_NaN == False:
              magnitude = grad.square().mean().sqrt()
              return grad * magnitude.clamp(max=args.clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
          return grad
  
      if args.diffusion_sampling_mode == 'ddim':
          sample_fn = diffusion.ddim_sample_loop_progressive
      else:
          sample_fn = diffusion.plms_sample_loop_progressive


      image_display = Output()
      for i in range(args.n_batches):
          if args.animation_mode == 'None':
            display.clear_output(wait=True)
            batchBar = tqdm(range(args.n_batches), desc ="Batches")
            batchBar.n = i
            batchBar.refresh()
          print('')
          display.display(image_display)
          gc.collect()
          torch.cuda.empty_cache()
          cur_t = diffusion.num_timesteps - skip_steps - 1
          total_steps = cur_t

          if perlin_init:
              init = regen_perlin(perlin_mode=perlin_mode, device=device, batch_size=batch_size)

          if args.diffusion_sampling_mode == 'ddim':
              samples = sample_fn(
                  model,
                  (batch_size, 3, args.side_y, args.side_x),
                  clip_denoised=clip_denoised,
                  model_kwargs={},
                  cond_fn=cond_fn,
                  progress=True,
                  skip_timesteps=skip_steps,
                  init_image=init,
                  randomize_class=randomize_class,
                  eta=eta,
              )
          else:
              samples = sample_fn(
                  model,
                  (batch_size, 3, args.side_y, args.side_x),
                  clip_denoised=clip_denoised,
                  model_kwargs={},
                  cond_fn=cond_fn,
                  progress=True,
                  skip_timesteps=skip_steps,
                  init_image=init,
                  randomize_class=randomize_class,
                  order=2,
              )
          
          
          # with run_display:
          # display.clear_output(wait=True)
          for j, sample in enumerate(samples):    
            cur_t -= 1
            intermediateStep = False
            if args.steps_per_checkpoint is not None:
                if j % steps_per_checkpoint == 0 and j > 0:
                  intermediateStep = True
            elif j in args.intermediate_saves:
              intermediateStep = True
            with image_display:
              if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                  for k, image in enumerate(sample['pred_xstart']):
                      # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                      current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                      percent = math.ceil(j/total_steps*100)
                      if args.n_batches > 0:
                        #if intermediates are saved to the subfolder, don't append a step or percentage to the name
                        if cur_t == -1 and args.intermediates_in_subfolder is True:
                          save_num = f'{frame_num:04}' if animation_mode != "None" else i
                          filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                        else:
                          #If we're working with percentages, append it
                          if args.steps_per_checkpoint is not None:
                            filename = f'{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png'
                          # Or else, iIf we're working with specific steps, append those
                          else:
                            filename = f'{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png'
                      image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                      if j % args.display_rate == 0 or cur_t == -1:
                        # image.save('progress.png')
                        image.save(f'{batchFolder}/progress.png')
                        # prints output on console.
                        if(console_preview): 
                          output = climage.convert(f'{batchFolder}/progress.png', width=console_preview_width)
                          print(output)
                        # display.clear_output(wait=True)
                        # display.display(display.Image('progress.png'))
                      if args.steps_per_checkpoint is not None:
                        if j % args.steps_per_checkpoint == 0 and j > 0:
                          if args.intermediates_in_subfolder is True:
                            image.save(f'{partialFolder}/{filename}')
                          else:
                            image.save(f'{batchFolder}/{filename}')
                      else:
                        if j in args.intermediate_saves:
                          if args.intermediates_in_subfolder is True:
                            image.save(f'{partialFolder}/{filename}')
                          else:
                            image.save(f'{batchFolder}/{filename}')
                      if cur_t == -1:
                        if frame_num == 0:
                          save_settings(
                            batchFolder = batchFolder,
                            batch_name = batch_name,
                            batchNum = batchNum,
                            setting_list={
                              'text_prompts': text_prompts,
                              'image_prompts': image_prompts,
                              'clip_guidance_scale': clip_guidance_scale,
                              'tv_scale': tv_scale,
                              'range_scale': range_scale,
                              'sat_scale': sat_scale,
                              # 'cutn': cutn,
                              'cutn_batches': cutn_batches,
                              'max_frames': max_frames,
                              'interp_spline': interp_spline,
                              # 'rotation_per_frame': rotation_per_frame,
                              'init_image': init_image,
                              'init_scale': init_scale,
                              'skip_steps': skip_steps,
                              # 'zoom_per_frame': zoom_per_frame,
                              'frames_scale': frames_scale,
                              'frames_skip_steps': frames_skip_steps,
                              'perlin_init': perlin_init,
                              'perlin_mode': perlin_mode,
                              'skip_augs': skip_augs,
                              'randomize_class': randomize_class,
                              'clip_denoised': clip_denoised,
                              'clamp_grad': clamp_grad,
                              'clamp_max': clamp_max,
                              'seed': seed,
                              'fuzzy_prompt': fuzzy_prompt,
                              'rand_mag': rand_mag,
                              'eta': eta,
                              'width': width_height[0],
                              'height': width_height[1],
                              'diffusion_model': diffusion_model,
                              'use_secondary_model': use_secondary_model,
                              'steps': steps,
                              'diffusion_steps': diffusion_steps,
                              'diffusion_sampling_mode': diffusion_sampling_mode,
                              'ViTB32': ViTB32,
                              'ViTB16': ViTB16,
                              'ViTL14': ViTL14,
                              'RN101': RN101,
                              'RN50': RN50,
                              'RN50x4': RN50x4,
                              'RN50x16': RN50x16,
                              'RN50x64': RN50x64,
                              'cut_overview': str(cut_overview),
                              'cut_innercut': str(cut_innercut),
                              'cut_ic_pow': cut_ic_pow,
                              'cut_icgray_p': str(cut_icgray_p),
                              'key_frames': key_frames,
                              'max_frames': max_frames,
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
                              'video_init_path':video_init_path,
                              'extract_nth_frame':extract_nth_frame,
                              'video_init_seed_continuity': video_init_seed_continuity,
                              'turbo_mode':turbo_mode,
                              'turbo_steps':turbo_steps,
                              'turbo_preroll':turbo_preroll,
                          })
                        if args.animation_mode != "None":
                          image.save('prevFrame.png')
                        image.save(f'{batchFolder}/{filename}')
                        if args.animation_mode == "3D":
                          # If turbo, save a blended image
                          if turbo_mode and frame_num > 0:
                            # Mix new image with prevFrameScaled
                            blend_factor = (1)/int(turbo_steps)
                            newFrame = cv2.imread('prevFrame.png') # This is already updated..
                            prev_frame_warped = cv2.imread('prevFrameScaled.png')
                            blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                            cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                          else:
                            image.save(f'{batchFolder}/{filename}')

                          if vr_mode:
                            generate_eye_views(TRANSLATION_SCALE, batchFolder, filename, frame_num, midas_model, midas_transform, vr_eye_angle=vr_eye_angle, vr_ipd=vr_ipd, device=device, args=args)

                        # if frame_num != args.max_frames-1:
                        #   display.clear_output()
          
          plt.plot(np.array(loss_values), 'r')

diffusion_model = "512x512_diffusion_uncond_finetune_008100" #@param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100"]
use_secondary_model = True #@param {type: 'boolean'}
diffusion_sampling_mode = 'ddim' #@param ['plms','ddim']  

use_checkpoint = True #@param {type: 'boolean'}
ViTB32 = True #@param{type:"boolean"}
ViTB16 = True #@param{type:"boolean"}
ViTL14 = False #@param{type:"boolean"}
RN101 = False #@param{type:"boolean"}
RN50 = True #@param{type:"boolean"}
RN50x4 = False #@param{type:"boolean"}
RN50x16 = False #@param{type:"boolean"}
RN50x64 = False #@param{type:"boolean"}

check_model_SHA = False #@param{type:"boolean"}

model_256_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
model_secondary_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'

model_256_link = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
model_512_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'
model_secondary_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'

model_256_path = f'{model_path}/256x256_diffusion_uncond.pt'
model_512_path = f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt'
model_secondary_path = f'{model_path}/secondary_model_imagenet_2.pth'

# Override Notebook defaults if external parameters were provided.
for param in ["diffusion_model", "use_secondary_model", "ViTB32", "ViTB16", "ViTL14", 
              "RN101", "RN50", "RN50x4", "RN50x64", "check_model_SHA"]:
  globals()[param]=get_param(param,globals()[param])

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

if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'{model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
    secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32 is True: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)) 
if ViTB16 is True: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device) ) 
if ViTL14 is True: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device) ) 
if RN50 is True: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x4 is True: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device)) 
if RN50x16 is True: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device)) 
if RN50x64 is True: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device)) 
if RN101 is True: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device)) 

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)


# Basic Settings:
batch_name = 'TimeToDisco' #@param{type: 'string'}
steps = 250 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
width_height = [1280, 768]#@param{type: 'raw'}
clip_guidance_scale = 5000 #@param{type: 'number'}
tv_scale =  0#@param{type: 'number'}
range_scale =   150#@param{type: 'number'}
sat_scale =   0#@param{type: 'number'}
cutn_batches = 4  #@param{type: 'number'}
skip_augs = False#@param{type: 'boolean'}

# Override Notebook defaults if external parameters were provided.
for param in ["batch_name", "steps", "width_height", "clip_guidance_scale", "tv_scale", 
              "range_scale", "sat_scale", "cutn_batches", "skip_augs"]:
  globals()[param]=get_param(param,globals()[param])

# **Init Settings:**
init_image = None #@param{type: 'string'}
init_scale = 1000 #@param{type: 'integer'}
skip_steps = 10 #@param{type: 'integer'}

# Make sure you set skip_steps to ~50% of your steps if you want to use an init image.

# Override Notebook defaults if external parameters were provided.
for param in ["init_image", "init_scale", "skip_steps"]:
  globals()[param]=get_param(param,globals()[param])
  
# Get corrected sizes
side_x = (width_height[0]//64)*64;
side_y = (width_height[1]//64)*64;
if side_x != width_height[0] or side_y != width_height[1]:
  print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

# Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})

# Make folder for batch
batchFolder = f'{outDirPath}/{batch_name}'
createPath(batchFolder)

# Animation Mode:
animation_mode = 'None' #@param ['None', '2D', '3D', 'Video Input'] {type:'string'}

# Override Notebook defaults if external parameters were provided.
for param in ["animation_mode"]:
  globals()[param]=get_param(param,globals()[param])
  
# For animation, you probably want to turn `cutn_batches` to 1 to make it quicker.
# Video Input Settings:

video_init_path = "training.mp4" #@param {type: 'string'}
extract_nth_frame = 2 #@param {type: 'number'}
video_init_seed_continuity = True #@param {type: 'boolean'}

# Override Notebook defaults if external parameters were provided.
for param in ["extract_nth_frame", "video_init_seed_continuity"]:
  globals()[param]=get_param(param,globals()[param])
  
if animation_mode == "Video Input":
  videoFramesFolder = f'videoFrames'
  createPath(videoFramesFolder)
  print(f"Exporting Video Frames (1 every {extract_nth_frame})...")
  try:
    for f in pathlib.Path(f'{videoFramesFolder}').glob('*.jpg'):
      f.unlink()
  except:
    print('')
  vf = f'select=not(mod(n\,{extract_nth_frame}))'
  subprocess.run(['ffmpeg', '-i', f'{video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{videoFramesFolder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
  #!ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg


# 2D Animation Settings:
# zoom is a multiplier of dimensions, 1 is no zoom.
# All rotations are provided in degrees.

key_frames = True #@param {type:"boolean"}
max_frames = 10000#@param {type:"number"}

# Override Notebook defaults if external parameters were provided.
for param in ["key_frames", "max_frames"]:
  globals()[param]=get_param(param,globals()[param])

if animation_mode == "Video Input":
  max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))

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

# Override Notebook defaults if external parameters were provided.
for param in ["interp_spline", "angle", "zoom", "translation_x", "translation_y", "translation_z", "rotation_3d_x", "rotation_3d_y",
  "rotation_3d_z", "midas_depth_model", "midas_weight", "near_plane", "far_plane", "fov", "padding_mode", "sampling_mode"]:
  globals()[param]=get_param(param,globals()[param])
  
# TURBO MODE
# Turbo Mode (3D anim only):
# (Starts after frame 10,) skips diffusion steps and just uses depth map to warp images for skipped frames.
# Speeds up rendering by 2x-4x, and may improve image coherence between frames. frame_blend_mode smooths abrupt texture changes across 2 frames.
# For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

turbo_mode = False #@param {type:"boolean"}
turbo_steps = "3" #@param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 10 # frames

# Override Notebook defaults if external parameters were provided.
for param in ["turbo_mode", "turbo_steps", "turbo_preroll"]:
  globals()[param]=get_param(param,globals()[param])

#insist turbo be used only w 3d anim.
if turbo_mode and animation_mode != '3D':
  print('=====')
  print('Turbo mode only available with 3D animations. Disabling Turbo.')
  print('=====')
  turbo_mode = False

# Coherency Settings:**
# 'frame_scale' tries to guide the new frame to looking like the old one. A good default is 1500.
frames_scale = 1500 #@param{type: 'integer'}

# 'frame_skip_steps' will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
frames_skip_steps = '60%' #@param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}

# Override Notebook defaults if external parameters were provided.
for param in ["frames_scale", "frames_skip_steps"]:
  globals()[param]=get_param(param,globals()[param])
  
# VR MODE
# VR Mode (3D anim only):
# Enables stereo rendering of left/right eye views (supporting Turbo) which use a different (fish-eye) camera projection matrix.   
# Note the images you're prompting will work better if they have some inherent wide-angle aspect
# The generated images will need to be combined into left/right videos. These can then be stitched into the VR180 format.
# Google made the VR180 Creator tool but subsequently stopped supporting it. It's available for download in a few places including https://www.patrickgrunwald.de/vr180-creator-download
# The tool is not only good for stitching (videos and photos) but also for adding the correct metadata into existing videos, which is needed for services like YouTube to identify the format correctly.
# Watching YouTube VR videos isn't necessarily the easiest depending on your headset. For instance Oculus have a dedicated media studio and store which makes the files easier to access on a Quest https://creator.oculus.com/manage/mediastudio/
# The command to get ffmpeg to concat your frames for each eye is in the form: `ffmpeg -framerate 15 -i frame_%4d_l.png l.mp4` (repeat for r)

vr_mode = False #@param {type:"boolean"}

# 'vr_eye_angle' is the y-axis rotation of the eyes towards the center
vr_eye_angle = 0.5 #@param{type:"number"}

# interpupillary distance (between the eyes)
vr_ipd = 5.0 #@param{type:"number"}

# Override Notebook defaults if external parameters were provided.
for param in ["vr_mode", "vr_eye_angle", "vr_ipd"]:
  globals()[param]=get_param(param,globals()[param])

#insist turbo be used only w 3d anim.
if vr_mode and animation_mode != '3D':
  print('=====')
  print('VR mode only available with 3D animations. Disabling VR.')
  print('=====')
  turbo_mode = False

if key_frames:
    try:
        angle_series = get_inbetweens(parse_key_frames(angle), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `angle` correctly for key frames.\n"
            "Attempting to interpret `angle` as "
            f'"0: ({angle})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        angle = f"0: ({angle})"
        angle_series = get_inbetweens(parse_key_frames(angle), max_frames=max_frames, interp_spline=interp_spline)

    try:
        zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `zoom` correctly for key frames.\n"
            "Attempting to interpret `zoom` as "
            f'"0: ({zoom})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        zoom = f"0: ({zoom})"
        zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames=max_frames, interp_spline=interp_spline)

    try:
        translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_x` correctly for key frames.\n"
            "Attempting to interpret `translation_x` as "
            f'"0: ({translation_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_x = f"0: ({translation_x})"
        translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames=max_frames, interp_spline=interp_spline)

    try:
        translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_y` correctly for key frames.\n"
            "Attempting to interpret `translation_y` as "
            f'"0: ({translation_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_y = f"0: ({translation_y})"
        translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames=max_frames, interp_spline=interp_spline)

    try:
        translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_z` correctly for key frames.\n"
            "Attempting to interpret `translation_z` as "
            f'"0: ({translation_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_z = f"0: ({translation_z})"
        translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames=max_frames, interp_spline=interp_spline)

    try:
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_x` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_x` as "
            f'"0: ({rotation_3d_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_x = f"0: ({rotation_3d_x})"
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames=max_frames, interp_spline=interp_spline)

    try:
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_y` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_y` as "
            f'"0: ({rotation_3d_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_y = f"0: ({rotation_3d_y})"
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames=max_frames, interp_spline=interp_spline)

    try:
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_z` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_z` as "
            f'"0: ({rotation_3d_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_z = f"0: ({rotation_3d_z})"
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames=max_frames, interp_spline=interp_spline)

else:
    angle = float(angle)
    zoom = float(zoom)
    translation_x = float(translation_x)
    translation_y = float(translation_y)
    translation_z = float(translation_z)
    rotation_3d_x = float(rotation_3d_x)
    rotation_3d_y = float(rotation_3d_y)
    rotation_3d_z = float(rotation_3d_z)


# Saving:

# Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps 
# A value of `2` will save a copy at 33% and 66%. 0 will save none.
# A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. (Make sure to include the brackets)

intermediate_saves = 0#@param{type: 'raw'}
intermediates_in_subfolder = True #@param{type: 'boolean'}

# Override Notebook defaults if external parameters were provided.
for param in ["intermediate_saves", "intermediates_in_subfolder"]:
  globals()[param]=get_param(param,globals()[param])

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

# Advanced Settings:
# There are a few extra advanced settings available if you double click this cell.
# Perlin init will replace your init, so uncheck if using one.

perlin_init = False  #@param{type: 'boolean'}
perlin_mode = 'mixed' #@param ['mixed', 'color', 'gray']
set_seed = 'random_seed' #@param{type: 'string'}
eta = 0.8#@param{type: 'number'}
clamp_grad = True #@param{type: 'boolean'}
clamp_max = 0.05 #@param{type: 'number'}

### EXTRA ADVANCED SETTINGS:
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05

# Cutn Scheduling:
# Format: '[40]*400+[20]*600' = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000
# 'cut_overview' and 'cut_innercut' are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure,
# innercuts are your standard cutn.

cut_overview = "[12]*400+[4]*600" #@param {type: 'string'}       
cut_innercut ="[4]*400+[12]*600"#@param {type: 'string'}  
cut_ic_pow = 1#@param {type: 'number'}  
cut_icgray_p = "[0.2]*400+[0]*600"#@param {type: 'string'}

# Override Notebook defaults if external parameters were provided.
for param in ["perlin_init", "perlin_mode", "set_seed", "eta", "clamp_grad", "clamp_max",
              "randomize_class", "clip_denoised", "fuzzy_prompt", "rand_mag",
              "cut_overview", "cut_innercut", "cut_ic_pow", "cut_icgray_p"]:
  globals()[param]=get_param(param,globals()[param])
  
# Prompts
# 'animation_mode' = None will only use the first set. 
# 'animation_mode' = '2D / Video' will run through them per the set frames and hold on the last one.

text_prompts = {
    0: ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.", "yellow color scheme"],
    100: ["This set of prompts start at frame 100","This prompt has weight five:5"],
}

image_prompts = {
    # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
}

# Override Notebook defaults if external parameters were provided.
for param in ["text_prompts", "image_prompts"]:
  globals()[param]=get_param(param,globals()[param])

# Diffuse
# Do the Run!
# 'n_batches' (ignored with animation modes.)
display_rate =  50 #@param{type: 'number'}
n_batches =  50 #@param{type: 'number'}

# Override Notebook defaults if external parameters were provided.
for param in ["display_rate", "n_batches"]:
  globals()[param]=get_param(param,globals()[param])

# Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})

batch_size = 1 

resume_run = False #@param{type: 'boolean'}
run_to_resume = 'latest' #@param{type: 'string'}
resume_from_frame = 'latest' #@param{type: 'string'}
retain_overwritten_frames = False #@param{type: 'boolean'}

# Override Notebook defaults if external parameters were provided.
for param in ["resume_run", "run_to_resume", "resume_from_frame", "retain_overwritten_frames"]:
  globals()[param]=get_param(param,globals()[param])

if retain_overwritten_frames is True:
  retainFolder = f'{batchFolder}/retained'
  createPath(retainFolder)

skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

if steps <= calc_frames_skip_steps:
  sys.exit("ERROR: You can't skip more steps than your total steps")

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

print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')

if set_seed == 'random_seed':
    random.seed()
    seed = random.randint(0, 2**32)
    print(f'Using seed: {seed}')
else:
    seed = int(set_seed)

args = {
    'animation_mode' : animation_mode,
    'batchNum': batchNum,
    'prompts_series':split_prompts(text_prompts, max_frames = max_frames) if text_prompts else None,
    'image_prompts_series':split_prompts(image_prompts, max_frames = max_frames) if image_prompts else None,
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
    'angle_series':angle_series,
    'zoom_series':zoom_series,
    'translation_x_series':translation_x_series,
    'translation_y_series':translation_y_series,
    'translation_z_series':translation_z_series,
    'rotation_3d_x_series':rotation_3d_x_series,
    'rotation_3d_y_series':rotation_3d_y_series,
    'rotation_3d_z_series':rotation_3d_z_series,
    'frames_scale': frames_scale,
    'calc_frames_skip_steps': calc_frames_skip_steps,
    'skip_step_ratio': skip_step_ratio,
    'calc_frames_skip_steps': calc_frames_skip_steps,
    'text_prompts': text_prompts,
    'image_prompts': image_prompts,
    'cut_overview': eval(cut_overview),
    'cut_innercut': eval(cut_innercut),
    'cut_ic_pow': cut_ic_pow,
    'cut_icgray_p': eval(cut_icgray_p),
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
}

args = SimpleNamespace(**args)

print('Prepping model...')
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load(f'{model_path}/{diffusion_model}.pt', map_location='cpu'))
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

gc.collect()
torch.cuda.empty_cache()
try:
  print('Seed used:', seed)
  do_run()
except KeyboardInterrupt:
    print('Run interrupted by user.')
    pass
finally:
    gc.collect()
    torch.cuda.empty_cache()


# Create the video
# Video file will save in the same folder as your images.

skip_video_for_run_all = True #@param {type: 'boolean'}

if skip_video_for_run_all == True:
  print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')

else:
  
  latest_run = batchNum
  folder = batch_name #@param
  run = latest_run #@param
  final_frame = 'final_frame'

  init_frame = 1#@param {type:"number"} This is the frame where the video will start
  last_frame = final_frame#@param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
  fps = 12#@param {type:"number"}
  # view_video_in_cell = True #@param {type: 'boolean'}

  frames = []
  # tqdm.write('Generating video...')

  if last_frame == 'final_frame':
    last_frame = len(glob(batchFolder+f"/{folder}({run})_*.png"))
    print(f'Total frames: {last_frame}')

  image_path = f"{outDirPath}/{folder}/{folder}({run})_%04d.png"
  filepath = f"{outDirPath}/{folder}/{folder}({run}).mp4"

  cmd = [ 'ffmpeg','-y','-vcodec','png','-r',str(fps),'-start_number',str(init_frame),'-i',image_path,
      '-frames:v',str(last_frame+1),'-c:v','libx264','-vf',f'fps={fps}','-pix_fmt','yuv420p','-crf','17','-preset',
      'veryslow',filepath]

  process = subprocess.Popen(cmd, cwd=f'{batchFolder}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  if process.returncode != 0:
      print(stderr)
      raise RuntimeError(stderr)
  else:
      print("The video is ready and saved to the images folder")
      