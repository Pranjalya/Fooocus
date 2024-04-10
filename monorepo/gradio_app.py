import os
import copy
import time
import random
import torch
from urllib.parse import urlparse
from typing import Optional
import gradio as gr

import core
import ldm_patched.modules.model_management
import inpaint_worker
import numpy as np
import gradio_hijack as grh
from utils import erode_or_dilate, HWC3, apply_wildcards, apply_arrays, apply_style, remove_empty_str, resample_image
from expansion import safe_str, FooocusExpansion
from pipeline_utils import *

MODEL_DIR = ".cache"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file



def downloading_inpaint_models():
    load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
            model_dir=MODEL_DIR,
            file_name='fooocus_inpaint_head.pth'
        )
    head_file = os.path.join(MODEL_DIR, 'fooocus_inpaint_head.pth')
    patch_file = None

    load_file_from_url(
                url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
                model_dir=MODEL_DIR,
                file_name='inpaint_v26.fooocus.patch'
            )
    patch_file = os.path.join(MODEL_DIR, 'inpaint_v26.fooocus.patch')
    return head_file, patch_file


def download_models():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=os.path.join("./fooocus_expansion"),
        file_name='pytorch_model.bin'
    )
    load_file_from_url(
        url='https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors',
        model_dir=MODEL_DIR,
        file_name='sd_xl_turbo_1.0_fp16.safetensors'
    )


def load_model(filename, base_model_additional_loras):
    final_expansion = None

    model_base = core.StableDiffusionModel()
    model_base = core.load_model(filename)
    print(f'Base model loaded: {model_base.filename}')

    model_refiner = core.StableDiffusionModel(
        unet=model_base.unet,
        vae=model_base.vae,
        clip=model_base.clip,
        clip_vision=model_base.clip_vision,
        filename=model_base.filename
    )
    model_refiner.vae = None
    model_refiner.clip = None
    model_refiner.clip_vision = None

    model_base.refresh_loras(base_model_additional_loras)

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_vae = model_base.vae

    final_refiner_unet = model_refiner.unet_with_lora
    final_refiner_vae = model_refiner.vae

    if final_expansion is None:
        final_expansion = FooocusExpansion()

    prepare_text_encoder(final_clip, final_expansion, async_call=True)
    return final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip, final_expansion


# Functions
def load_inpaint_images(inpaint_input_image, inpaint_mask_image_upload, inpaint_erode_or_dilate, refiner_model_name="None"):
    inpaint_image = inpaint_input_image['image']
    inpaint_mask = inpaint_input_image['mask'][:, :, 0]

    if inpaint_mask_image_upload.ndim == 3:
        H, W, C = inpaint_image.shape
        inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
        inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
        inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
        inpaint_mask = np.maximum(inpaint_mask, inpaint_mask_image_upload)

    if int(inpaint_erode_or_dilate) != 0:
        inpaint_mask = erode_or_dilate(inpaint_mask, inpaint_erode_or_dilate)

    inpaint_image = HWC3(inpaint_image)
    if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
            and (np.any(inpaint_mask > 127)):
        # progressbar(async_task, 1, 'Downloading upscale models ...')
        # modules.config.downloading_upscale_model()
        print('Downloading inpainter ...')
        inpaint_head_model_path, inpaint_patch_model_path = downloading_inpaint_models()
        base_model_additional_loras = [(inpaint_patch_model_path, 1.0)]
        print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
        if refiner_model_name == 'None':
            use_synthetic_refiner = True
            refiner_switch = 0.8
    return inpaint_image, inpaint_mask, inpaint_head_model_path, inpaint_patch_model_path, \
        base_model_additional_loras, use_synthetic_refiner, refiner_switch



def expand_prompt(
    prompt,
    negative_prompt,
    image_number=2,
    base_model_name="",
    refiner_model_name="None",
    base_model_additional_loras=[],
    use_synthetic_refiner=True,
    seed=123456,
    style_selections=[],
    cfg_scale=7.0
):

    fooocus_expansion = "Fooocus V2"
    if fooocus_expansion in style_selections:
        use_expansion = True
        style_selections.remove(fooocus_expansion)
    else:
        use_expansion = False
    use_style = len(style_selections) > 0

    prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
    negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')

    prompt = prompts[0]
    negative_prompt = negative_prompts[0]

    if prompt == '':
        # disable expansion when empty since it is not meaningful and influences image prompt
        use_expansion = False

    extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
    extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

    print('Loading models ...')
    print("Models")
    print("refiner_model_name", refiner_model_name)
    print("base_model_name", base_model_name)
    print("loras", [])
    print("base_model_additional_loras", base_model_additional_loras)
    print("use_synthetic_refiner", use_synthetic_refiner)

    final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip, final_expansion = load_model(base_model_name, base_model_additional_loras)

    # pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
    #                             loras=[], base_model_additional_loras=base_model_additional_loras,
    #                             use_synthetic_refiner=use_synthetic_refiner)

    print('Processing prompts ...')
    tasks = []
    
    for i in range(image_number):
        task_seed = (seed + i) % (1234569)  # randint is inclusive, % is not

        task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
        task_prompt = apply_wildcards(prompt, task_rng, i, False)
        task_prompt = apply_arrays(task_prompt, i)
        task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, False)
        task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, False) for pmt in extra_positive_prompts]
        task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, False) for pmt in extra_negative_prompts]

        positive_basic_workloads = []
        negative_basic_workloads = []

        if use_style:
            for s in style_selections:
                p, n = apply_style(s, positive=task_prompt)
                positive_basic_workloads = positive_basic_workloads + p
                negative_basic_workloads = negative_basic_workloads + n
        else:
            positive_basic_workloads.append(task_prompt)

        negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

        positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
        negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

        positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
        negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

        tasks.append(dict(
            task_seed=task_seed,
            task_prompt=task_prompt,
            task_negative_prompt=task_negative_prompt,
            positive=positive_basic_workloads,
            negative=negative_basic_workloads,
            expansion='',
            c=None,
            uc=None,
            positive_top_k=len(positive_basic_workloads),
            negative_top_k=len(negative_basic_workloads),
            log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
            log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
        ))

    if use_expansion:
        for i, t in enumerate(tasks):
            print(f'Preparing Fooocus text #{i + 1} ...')
            expansion = final_expansion(t['task_prompt'], t['task_seed'])
            print(f'[Prompt Expansion] {expansion}')
            t['expansion'] = expansion
            t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

    for i, t in enumerate(tasks):
        print(f'Encoding positive #{i + 1} ...')
        t['c'] = clip_encode(final_clip, texts=t['positive'], pool_top_k=t['positive_top_k'])

    for i, t in enumerate(tasks):
        if abs(float(cfg_scale) - 1.0) < 1e-4:
            t['uc'] = clone_cond(t['c'])
        else:
            print (f'Encoding negative #{i + 1} ...')
            t['uc'] = clip_encode(final_clip, texts=t['negative'], pool_top_k=t['negative_top_k'])
    return tasks, final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip, final_expansion



def inpaint_image(
    inpaint_input_image,
    inpaint_mask_image,
    inpaint_erode_or_dilate,
    steps,
    refiner_switch,
    prompt="",
    negative_prompt="",
    width=1024,
    height=1024,
    sampler_name="dpmpp_2m_sde_gpu",
    scheduler_name="karras",
    style_selections=[],
    num_images=1,
    guidance_scale=7.0,
    inpaint_strength=1.0
):
    download_models()
    execution_start_time = time.perf_counter()
    inpaint_image, inpaint_mask, inpaint_head_model_path, inpaint_patch_model_path, \
        base_model_additional_loras, use_synthetic_refiner, refiner_switch = load_inpaint_images(inpaint_input_image, inpaint_mask_image, inpaint_erode_or_dilate, refiner_model_name="None")
    switch = int(round(steps * refiner_switch))
    print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
    print(f'[Parameters] Steps = {steps} - {switch}')

    tasks, final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip, final_expansion = expand_prompt(prompt, negative_prompt, num_images, base_model_name=".cache/sd_xl_turbo_1.0_fp16.safetensors", style_selections=style_selections, cfg_scale=guidance_scale)
    print(tasks)

    denoising_strength = inpaint_strength
    inpaint_respective_field = 0

    inpaint_worker.current_task = inpaint_worker.InpaintWorker(
        image=inpaint_image,
        mask=inpaint_mask,
        use_fill=denoising_strength > 0.99,
        k=inpaint_respective_field
    )

    inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
    inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
    inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)


    print('VAE Inpaint encoding ...')

    refiner_swap_method = "joint"

    candidate_vae, candidate_vae_swap = get_candidate_vae(
        steps=steps,
        switch=switch,
        denoise=denoising_strength,
        refiner_swap_method=refiner_swap_method,
        final_refiner_vae=final_refiner_vae,
        final_refiner_unet=final_refiner_unet,
        final_vae=final_vae
    )

    print("VAE Candidate:", candidate_vae, candidate_vae_swap)

    latent_inpaint, latent_mask = core.encode_vae_inpaint(
        mask=inpaint_pixel_mask,
        vae=candidate_vae,
        pixels=inpaint_pixel_image)

    latent_swap = None
    if candidate_vae_swap is not None:
        print('VAE SD15 encoding ...')
        latent_swap = core.encode_vae(
            vae=candidate_vae_swap,
            pixels=inpaint_pixel_fill)['samples']

    print('VAE encoding ...')
    latent_fill = core.encode_vae(
        vae=candidate_vae,
        pixels=inpaint_pixel_fill)['samples']

    inpaint_worker.current_task.load_latent(
        latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

    inpaint_parameterized = True
    if inpaint_parameterized:
        final_unet = inpaint_worker.current_task.patch(
            inpaint_head_model_path=inpaint_head_model_path,
            inpaint_latent=latent_inpaint,
            inpaint_latent_mask=latent_mask,
            model=final_unet
        )

    inpaint_disable_initial_latent = True
    initial_latent = None
    if not inpaint_disable_initial_latent:
        initial_latent = {'samples': latent_fill}

    B, C, H, W = latent_fill.shape
    height, width = H * 8, W * 8
    final_height, final_width = inpaint_worker.current_task.image.shape[:2]
    print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

    all_steps = steps * num_images

    print(f'[Parameters] Denoising Strength = {denoising_strength}')

    if isinstance(initial_latent, dict) and 'samples' in initial_latent:
        log_shape = initial_latent['samples'].shape
    else:
        log_shape = f'Image Space {(height, width)}'

    print(f'[Parameters] Initial Latent shape: {log_shape}')

    preparation_time = time.perf_counter() - execution_start_time
    print(f'Preparation time: {preparation_time:.2f} seconds')

    final_sampler_name = sampler_name
    final_scheduler_name = scheduler_name

    output_images = []

    for current_task_id, task in enumerate(tasks):
        execution_start_time = time.perf_counter()
        positive_cond, negative_cond = task['c'], task['uc']
        imgs = process_diffusion(
                positive_cond=positive_cond,
                negative_cond=negative_cond,
                steps=steps,
                switch=switch,
                width=width,
                height=height,
                image_seed=task['task_seed'],
                callback=None,
                sampler_name=final_sampler_name,
                scheduler_name=final_scheduler_name,
                latent=initial_latent,
                denoise=denoising_strength,
                tiled=False,
                cfg_scale=float(guidance_scale),
                refiner_swap_method=refiner_swap_method,
                disable_preview=True,
                final_unet=final_unet, 
                final_vae=final_vae, 
                final_refiner_unet=final_refiner_unet, 
                final_refiner_vae=final_refiner_vae, 
                final_clip=final_clip
            )
        del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

        if inpaint_worker.current_task is not None:
            imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]
        
        execution_time = time.perf_counter() - execution_start_time
        print(f'Generating and saving time: {execution_time:.2f} seconds')
        
        output_images.extend(imgs)

    return output_images



def trigger_inpaint(
    inpaint_input_image, 
    inpaint_mask_image,
    steps_count=30,
    refiner_switch=0.8,
    guidance_scale=7.0,
    sharpness=10.0,
    inpaint_strength=1.0,
    inpaint_respective_field=0,
    inpaint_erode_or_dilate=0,
    adaptive_cfg=4.0,
    sampler_name="dpmpp_2m_sde_gpu",
    scheduler_name="karras",
    style_selections=["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],
    prompt="bright",
    negative_prompt="",
    num_images=2,
):
    _ = inpaint_image(inpaint_input_image, inpaint_mask_image, inpaint_erode_or_dilate, steps_count, refiner_switch, prompt, negative_prompt, sampler_name=sampler_name, scheduler_name=scheduler_name, style_selections=style_selections, num_images=num_images, guidance_scale=guidance_scale, inpaint_strength=inpaint_strength)
    return []


with gr.Blocks() as demo:
    gr.Markdown("# Foooooooocus SDXL Turbo")

    with gr.Row():
        inpaint_input_image = grh.Image(label='Drag inpaint or outpaint image to here', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas')
        inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', height=500, visible=True)
    with gr.Row():
        steps_count = gr.Slider(minimum=1, maximum=200, value=30, step=1, label="Steps")
        refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='Use 0.4 for SD1.5 realistic models; '
                                                    'or 0.667 for SD1.5 anime models; '
                                                    'or 0.8 for XL-refiners; '
                                                    'or any value for switching two SDXL models.')
    with gr.Row():
        guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=7.0,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')
        sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001, value=10.0,
                                info='Higher value means image and texture are sharper.')
    with gr.Row():
        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                minimum=0.0, maximum=1.0, step=0.001, value=0,
                                                info='The area to inpaint. '
                                                    'Value 0 is same as "Only Masked" in A1111. '
                                                    'Value 1 is same as "Whole Image" in A1111. '
                                                    'Only used in inpaint, not used in outpaint. '
                                                    '(Outpaint always use 1.0)')
        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                            minimum=-64, maximum=64, step=1, value=0,
                                            info='Positive value will make white area in the mask larger, '
                                                    'negative value will make white area smaller.'
                                                    '(default is 0, always process before any mask invert)')
    with gr.Row():
        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=4.0,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
        sampler_name = gr.Dropdown(label='Sampler', choices=["dpmpp_2m_sde_gpu"],
                                    value="dpmpp_2m_sde_gpu")
        scheduler_name = gr.Dropdown(label='Scheduler', choices=["karras"],
                                        value="karras")
    with gr.Row():
        style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],
                                                    value=["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
    with gr.Row():
        prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        container=False, autofocus=True, elem_classes='type_row', lines=2, value="blue, beautiful")
        negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                             info='Describing what you do not want to see.', lines=2,
                                             elem_id='negative_prompt')
    with gr.Row():
        num_images = gr.Slider(1, 10, value=2, step=1, label="Number of images")
        generate = gr.Button(label="Generate Images")
    
    gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=1024,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery')

    generate.click(trigger_inpaint, [inpaint_input_image, inpaint_mask_image, steps_count, refiner_switch, guidance_scale, sharpness, inpaint_strength, inpaint_respective_field, inpaint_erode_or_dilate, adaptive_cfg, sampler_name, scheduler_name, style_selections, prompt, negative_prompt, num_images], gallery)


demo.launch(debug=True)