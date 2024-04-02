import gradio as gr
import gradio_hijack as grh
import os
import numpy as np
from utils import erode_or_dilate, HWC3


MODEL_DIR = ".cache"
os.makedirs(MODEL_DIR, exist_ok=True)



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
        model_dir=MODEL_DIR,
        file_name='pytorch_model.bin'
    )
    load_file_from_url(
        url='https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors',
        model_dir=MODEL_DIR,
        file_name='sd_xl_turbo_1.0_fp16.safetensors'
    )



# Functions
def load_inpaint_images(inpaint_input_image, inpaint_erode_or_dilate, refiner_model_name="None"):
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




with gr.Blocks() as demo:
    gr.Markdown("# Foooooooocus SDXL Turbo")

    with gr.Row():
        inpaint_input_image = grh.Image(label='Drag inpaint or outpaint image to here', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas')
        inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy', height=500, visible=False)
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
                                value=modules.config.default_sample_sharpness,
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

