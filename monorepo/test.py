sampler_name = "dpmpp_2m_sde_gpu"
scheduler_name = "karras"
steps = 30
refiner_switch = 0.8
switch = int(round(steps * refiner_switch))
refiner_model_name = "None"
adaptive_cfg = 7
sharpness = 10
controlnet_softness = 0.25
adm_scaler_positive = 1.5
adm_scaler_negative = 0.8
adm_scaler_end = 0.3
cfg_scale = 4.0

initial_latent = None
denoising_strength = 1.0 # inpaint_strength = 1.0
inpaint_respective_field = 0

width, height = 1024, 1024

style_selections = ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp']
use_style = True

inpaint_patch_model_path = "inpaint_v26.fooocus.patch"
base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]


def downloading_inpaint_models():
    load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
            model_dir=path_inpaint,
            file_name='fooocus_inpaint_head.pth'
        )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    load_file_from_url(
                url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
                model_dir=path_inpaint,
                file_name='inpaint_v26.fooocus.patch'
            )
    patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')


def download_models():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )
    load_file_from_url(
        url='https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors',
        model_dir=config.path_fooocus_expansion,
        file_name='sd_xl_turbo_1.0_fp16.safetensors'
    )


def load_inpaint_images():
    inpaint_image = inpaint_input_image['image']
    inpaint_mask = inpaint_input_image['mask'][:, :, 0]

    if inpaint_mask_upload_checkbox:
        if isinstance(inpaint_mask_image_upload, np.ndarray):
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
            and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
        progressbar(async_task, 1, 'Downloading upscale models ...')
        modules.config.downloading_upscale_model()
        if inpaint_parameterized:
            progressbar(async_task, 1, 'Downloading inpainter ...')
            inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                inpaint_engine)
            base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
            print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
            if refiner_model_name == 'None':
                use_synthetic_refiner = True
                refiner_switch = 0.8
        else:
            inpaint_head_model_path, inpaint_patch_model_path = None, None
            print(f'[Inpaint] Parameterized inpaint is disabled.')
        if inpaint_additional_prompt != '':
            if prompt == '':
                prompt = inpaint_additional_prompt
            else:
                prompt = inpaint_additional_prompt + '\n' + prompt
        goals.append('inpaint')


def expand_prompt():
    progressbar(async_task, 3, 'Processing prompts ...')
    tasks = []
    
    for i in range(image_number):
        task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not

        task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
        task_prompt = apply_wildcards(prompt, task_rng, i, read_wildcards_in_order)
        task_prompt = apply_arrays(task_prompt, i)
        task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, read_wildcards_in_order)
        task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in extra_positive_prompts]
        task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in extra_negative_prompts]

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
            progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
            expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
            print(f'[Prompt Expansion] {expansion}')
            t['expansion'] = expansion
            t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

    for i, t in enumerate(tasks):
        progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
        t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

    for i, t in enumerate(tasks):
        if abs(float(cfg_scale) - 1.0) < 1e-4:
            t['uc'] = pipeline.clone_cond(t['c'])
        else:
            progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
            t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])


def inpaint_goals(inpaint_strength=1.0):
    denoising_strength = inpaint_strength

    inpaint_worker.current_task = inpaint_worker.InpaintWorker(
        image=inpaint_image,
        mask=inpaint_mask,
        use_fill=denoising_strength > 0.99,
        k=inpaint_respective_field
    )

    progressbar(async_task, 13, 'VAE Inpaint encoding ...')

    inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
    inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
    inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

    candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
        steps=steps,
        switch=switch,
        denoise=denoising_strength,
        refiner_swap_method=refiner_swap_method
    )

    latent_inpaint, latent_mask = core.encode_vae_inpaint(
        mask=inpaint_pixel_mask,
        vae=candidate_vae,
        pixels=inpaint_pixel_image)

    latent_swap = None

    progressbar(async_task, 13, 'VAE encoding ...')
    latent_fill = core.encode_vae(
        vae=candidate_vae,
        pixels=inpaint_pixel_fill)['samples']

    inpaint_worker.current_task.load_latent(
        latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

    pipeline.final_unet = inpaint_worker.current_task.patch(
        inpaint_head_model_path=inpaint_head_model_path,
        inpaint_latent=latent_inpaint,
        inpaint_latent_mask=latent_mask,
        model=pipeline.final_unet
    )

    inpaint_disable_initial_latent = True
    if not inpaint_disable_initial_latent:
        initial_latent = {'samples': latent_fill}

    B, C, H, W = latent_fill.shape
    height, width = H * 8, W * 8
    final_height, final_width = inpaint_worker.current_task.image.shape[:2]
    print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')




def generate_image():
    all_steps = steps * image_number
    print(f'[Parameters] Denoising Strength = {denoising_strength}')
    
    log_shape = f'Image Space {(height, width)}'

    print(f'[Parameters] Initial Latent shape: {log_shape}')

    preparation_time = time.perf_counter() - execution_start_time
    print(f'Preparation time: {preparation_time:.2f} seconds')

    final_sampler_name = sampler_name
    final_scheduler_name = scheduler_name

    async_task.yields.append(['preview', (13, 'Moving model to GPU ...', None)])

    imgs = pipeline.process_diffusion(
        positive_cond=positive_cond,
        negative_cond=negative_cond,
        steps=steps,
        switch=switch,
        width=width,
        height=height,
        image_seed=task['task_seed'],
        callback=callback,
        sampler_name=final_sampler_name,
        scheduler_name=final_scheduler_name,
        latent=initial_latent,
        denoise=denoising_strength,
        tiled=tiled,
        cfg_scale=cfg_scale,
        refiner_swap_method=refiner_swap_method,
        disable_preview=disable_preview
    )

    del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

    if inpaint_worker.current_task is not None:
        imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

    img_paths = []
    for x in imgs:
        d = [('Prompt', 'prompt', task['log_positive_prompt']),
                ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                ('Fooocus V2 Expansion', 'prompt_expansion', task['expansion']),
                ('Styles', 'styles', str(raw_style_selections)),
                ('Performance', 'performance', performance_selection.value)]

        if performance_selection.steps() != steps:
            d.append(('Steps', 'steps', steps))

        d += [('Resolution', 'resolution', str((width, height))),
                ('Guidance Scale', 'guidance_scale', guidance_scale),
                ('Sharpness', 'sharpness', sharpness),
                ('ADM Guidance', 'adm_guidance', str((
                    modules.patch.patch_settings[pid].positive_adm_scale,
                    modules.patch.patch_settings[pid].negative_adm_scale,
                    modules.patch.patch_settings[pid].adm_scaler_end))),
                ('Base Model', 'base_model', base_model_name),
                ('Refiner Model', 'refiner_model', refiner_model_name),
                ('Refiner Switch', 'refiner_switch', refiner_switch)]

        if refiner_model_name != 'None':
            if overwrite_switch > 0:
                d.append(('Overwrite Switch', 'overwrite_switch', overwrite_switch))
            if refiner_swap_method != flags.refiner_swap_method:
                d.append(('Refiner Swap Method', 'refiner_swap_method', refiner_swap_method))
        if modules.patch.patch_settings[pid].adaptive_cfg != modules.config.default_cfg_tsnr:
            d.append(('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[pid].adaptive_cfg))

        d.append(('Sampler', 'sampler', sampler_name))
        d.append(('Scheduler', 'scheduler', scheduler_name))
        d.append(('Seed', 'seed', str(task['task_seed'])))

        if freeu_enabled:
            d.append(('FreeU', 'freeu', str((freeu_b1, freeu_b2, freeu_s1, freeu_s2))))

        for li, (n, w) in enumerate(loras):
            if n != 'None':
                d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))

        metadata_parser = None
        if save_metadata_to_images:
            metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
            metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                                        task['log_negative_prompt'], task['negative'],
                                        steps, base_model_name, refiner_model_name, loras)
        d.append(('Metadata Scheme', 'metadata_scheme', metadata_scheme.value if save_metadata_to_images else save_metadata_to_images))
        d.append(('Version', 'version', 'Fooocus v' + fooocus_version.version))
        img_paths.append(log(x, d, metadata_parser, output_format))

    yield_result(async_task, img_paths, do_not_show_finished_images=len(tasks) == 1 or disable_intermediate_results)