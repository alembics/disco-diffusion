# Parameters

Credits to Zippy for using his [Zippy's Disco Diffusion Cheatsheet](https://docs.google.com/document/d/1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g) as a base of information for this documentation.

## `text_prompts`
  
In DD, `text_prompts` can be a few words, a long sentence, or a few sentences. Writing prompts is an art in and of itself that won't be covered here, but the DD prompts section has some examples including the formatting required.

**Default value:**

```json
{
    "0": [
        "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.",
        "yellow color scheme"
    ],
    "100": [
        "This set of prompts start at frame 100",
        "This prompt has weight five:5"
    ]
}
```
### **Advanced topic: Prompt weights and animation frames**

You can run a successful batch of images or an animation with a single text prompt.  However, DD allows additional flexibility in the prompt section.
 
Prompts are broken down into an animation frame number and a list of strings. The list of strings are individual prompts that the diffusion process will try to drive the image toward. The end of the string can contain a `":num"` value to indicate the weight of that prompt relative to the other prompts. 
 
Each prompt's relative contribution in driving the diffusion direction is its weight divided by the sum of all the weights. Weights can be negative! Negative weights can help inhibit features that match an undesired prompt, such as text, watermarks, or other undesired features.  (e.g. `0:["rocky beach:2", "sky:-1"]` will push the image towards a rocky beach while attenuating sky details. Important: the sum of the weights must not equal 0.)
 
The number at the very beginning of the line is an animation frame number.  If animation is used, you can change prompts over time by adding more lines of prompts with different animation frames, and DD will switch text prompts at the specified frame. Note: adding multiple prompts in this manner only works with animations.  If you are running a batch of individual images, DD will just use the first text prompt. 

## `image_prompts`

Source images are another way to guide the diffusion process toward a look or style.  Using image prompts to create other images is an indirect method, however, and not as reliable as text prompts.

**Default value:** `None`

## `batch_name`

File and folder name for the batch. Final images and/or videos will be saved in `images_out/[batch_name]`

**Default value:** `TimeToDisco`

## `width_height`

Final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of `64px` in your dimensions, DD will adjust the dimensions of your image to make it so.

In the DD context, on a typical Colab system, [512x768] is a medium image and is a good starting point.  [1024x768] is considered a large image, and may cause an OOM (out of memory) error.
  
Significantly larger dimensions will use significantly more memory (and may crash DD!) so start small at first.  If you want a very large final image, a common practice is to generate medium sized images using DD, then to use a separate AI "upscaler" to increase the dimensions of the DD-produced image.
 
One interesting CLIP-Diffusion phenomenon is that if you make the image very tall in dimension (ie. `512 x 1024`), then you can get better results for tall/giant prompt subjects; e.g. “A giant creature.” Wide images can also be used for wide subjects like panoramic landscapes. This is likely due to the orientation and format of original images used to train the various CLIP datasets.

**Default value:** `[1280,768]`

## `steps`

When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration) involves the AI looking at subsets of the image called "cuts" and calculating the "direction" the image should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser, and moves to the next step.
 
Increasing steps will provide more opportunities for the AI to adjust the image, and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps comes at the expense of longer render times.  Also, while increasing steps should generally increase image quality, there is a diminishing return on additional steps beyond `250 - 500` steps.  However, some intricate images can take `1000`, `2000`, or more steps.  It is really up to the user.
 
Just know that the render time is directly related to the number of steps, and many other parameters have a major impact on image quality, without costing additional time.

**Default value:** `250`

## `skip_steps`

Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high, so images change dramatically in early steps. 

![Diffusion De-Noising Curve Chart](/docs/images/dd_curve.png)
 
As DD moves along the curve, noise levels (and thus the amount an image changes per step) declines, and image coherence from one step to the next increases.
 
The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be skipped without affecting the final image. You can experiment with this as a way to cut render times.
  
  - *Note:*
    
    If you skip too many steps, however, the remaining noise may not be high enough to generate new content, and thus may not have ‘time left’ to finish an image satisfactorily.

Also, depending on your other settings, you may need to skip steps to prevent CLIP from overshooting your goal, resulting in ‘blown out’ colors (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate other problems. 
 
Lastly, if using an `init_image`, you will need to skip `~50%` of the diffusion steps to retain the shapes in the original init image. 
 
However, if you’re using an `init_image`, you can also adjust skip_steps up or down for creative reasons.  With low `skip_steps` you can get a result "inspired by" the init_image which will retain the colors and rough layout and shapes but look quite different. With high `skip_steps` you can preserve most of the `init_image/` contents and just do fine tuning of the texture.

**Default value:** `10`

## `clip_guidance_scale`

CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust CGS. 
 
Note that this parameter generally scales with image dimensions. In other words, if you increase your total dimensions by `50%` (e.g. a change from `512 x 512` to `512 x 768`), then to maintain the same effect on the image, you’d want to increase `clip_guidance_scale` from `5000` to `7500`.
 

**Default value:** `5000`

**Valid Range:** `1500-100000`

## `tv_scale`
Total variance denoising. Optional, set to zero to turn off. Controls ‘smoothness’ of final output. If used, tv_scale will try to smooth out your final image to reduce overall noise. If your image is too ‘crunchy’, increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.

See https://en.wikipedia.org/wiki/Total_variation_denoising

**Default value:** `0`

**Valid Range:** `0-1000`

## `range_scale`

Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images. Higher range_scale will reduce contrast, for more muted images.

**Default value:** `150`

**Valid Range:** `0-1000`

## `sat_scale`
Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation. If your image is too saturated, increase `sat_scale` to reduce the saturation.

**Default value:** `0`

**Valid Range:** `0-2000`

## `init_image`

Optional. Recall that in the image sequence above, the first image shown is just noise.  If an `init_image` is provided, diffusion will replace the noise with the `init_image` as its starting state.  To use an `init_image`, place the image in your `init_images/` folder. 
 
If using an `init_image`, you may need to increase `skip_step`s to `~50%` of total steps to retain the character of the init. See `skip_steps` above for further discussion.
 
## `init_scale`

This controls how strongly CLIP will try to match the `init_image` provided.  This is balanced against the `clip_guidance_scale` (CGS) above.  Too much init scale, and the image won’t change much during diffusion. Too much CGS and the init image will be lost.

**Default value:** `1000`

**Valid Range:** `10-20000`
 
## `cutn_batches`
Each iteration, the AI cuts the image into smaller pieces known as **cuts**, and compares each **cut** to the prompt to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has more chances to fine-tune the image precision in each timestep.
 
**Additional cuts are memory intensive, however.**  If DD tries to evaluate too many cuts at once, it can run out of memory.  You can use `cutn_batches` to increase cuts per timestep without increasing memory usage.
 
At the default settings, DD is scheduled to do 16 cuts per timestep.  If `cutn_batches` is set to `1`, there will indeed only be `16` cuts total per timestep. 
 
However, if `cutn_batches` is increased to `4`, DD will do `64` cuts total in each timestep, divided into 4 sequential batches of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts, but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take **~4 times as long** to render each image.
 
So, (`scheduled cuts`) x (`cutn_batches`) = **total cuts per timestep**. Increasing `cutn_batches` will increase render times, however, as the work is being done sequentially.  DD's default cut schedule is a good place to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.

**Default value:** `4`

**Valid Range:** `1-8`

## `skip_augs`

As part of its code, DD has some "torchvision augmentations" which introduce randomized image scaling, perspective and other selective adjustments during image creation. These augmentations are intended to help improve image quality, but can have a "smoothing" effect on edges that you may not want. By setting `skip_augs` to true, you can skip these augmentations and speed up your renders slightly. Suggest you experiment with this setting to understand how it affects your projects.

**Default value:** `false`

## `n_batches`

This variable sets the number of still images you want DD to create.  If you are using an `animation_mode`, DD will ignore `n_batches` and create a single set of animated frames based on the animation settings.

**Default value:** `50`

**Valid Range:** `1-100`

## `display_rate`

During a diffusion run, you can monitor the progress of each image being created with this variable.  If `display_rate` is set to `50`, DD will show you the in-progress image every 50 timesteps.  
 
Setting this to a lower value, like `5` or `10`, is a good way to get an early peek at where your image is heading. If you don’t like the progression, just interrupt execution, change some settings, and re-run.  If you are planning a long, unmonitored batch, it's better to set `display_rate` equal to steps, because displaying interim images does slow processing down slightly.

**Default value:** `50`

**Valid Range:** `5-500`

## `resume_run`

If your batch run gets interrupted (either because you stopped it, or because of a disconnection,) you can resume your batch run where you left off using this checkbox. However, **you MUST not change the settings in the batch**, or it cannot reliably be resumed.  Other parameters (`run_to_resume`, `resume_from_frame`, `retain_overwritten_frames`)` control how you want the batch to be resumed.
 
If you have interrupted a run and tweaked settings, you should NOT use `resume_run`, as this would be considered a new run with the new settings.

**Default value:** `false`

## `diffusion_model`

Diffusion model of choice

**Default value:** `512x512_diffusion_uncond_finetune_008100`

**Valid Options:**
  - `256x256_diffusion_uncond`
  - `512x512_diffusion_uncond_finetune_008100`

## `use_secondary_model`

Option to use a secondary purpose-made diffusion model to clean up interim diffusion images for CLIP evaluation.  If this option is turned off, DD will use the regular (large) diffusion model.  Using the secondary model is faster - one user reported a 50% improvement in render speed.  However, the secondary model is much smaller, and may reduce image quality and detail.  It is suggested that you experiment with this.

**Default value:** `true`
 
## `sampling_mode`

Two alternate diffusion denoising algorithms.  `ddim` has been around longer, and is more established and tested.  `plms` is a newly added alternate method that promises good diffusion results in fewer steps, but has not been as fully tested and may have side effects. This new plms mode is actively being researched in the #settings-and-techniques channel in the DD Discord.

**Default value:** `ddim`

**Valid Options:**
  - `ddim`
  - `plms`
 
## `timestep_respacing`

This is an internal variable that you should leave alone.  In future DD releases, this will likely be hidden from users, as it's not meant to be edited directly.
 
## `diffusion_steps`

This is an internal variable that you should leave alone.  In future DD releases, this will likely be hidden from users, as it's not meant to be edited directly.
 
## `use_checkpoint`

This option helps save VRAM while generating images. If you are on a very powerful machine (e.g. A100) you may be able to turn this option off and speed things up. However, you might also immediately encounter a CUDA OOM error, so use caution.

**Default value:** `true`
 
## `CLIP Model selectors`

These various CLIP models are available for you to use during image generation.  Models have different styles or "flavors", so look around.  
 
You can mix in multiple models as well for different results.  However, keep in mind that some models are extremely memory-hungry, and turning on additional models will take additional memory and may cause a crash.
 
Default Values and the rough order of speed/mem usage is (smallest/fastest to largest/slowest):

| Model | Default Value |
| ----- | ------------- |
| `ViTB32` | `true` |
| `RN50` | `true` |
| `RN101` | `false` |
| `ViTB16`| `true` |
| `RN50x4`| `false` |
| `RN50x16`| `false` |
| `RN50x64`| `false` |
| `ViTL14`| `false` |

**Notes:**

- For RN50x64 & ViTL14 you may need to use fewer cuts, depending on your VRAM.
- If you change any of the Diffusion and CLIP model settings in this section, you should restart your Python or Notebook process and run from the beginning.
 
## `intermediate_saves`

In addition to a final image, DD can save intermediate images from partway through the diffusion curve.  This is helpful to diagnose image problems, or if you want to make a timeline or video of the diffusion process itself. See the notebook for instructions on using this.

**Default value:** `0`

## `intermediates_in_subfolder`

If saving intermediate images, this option will store intermediate images in a subfolder called 'partials'.

**Default value:** `true`