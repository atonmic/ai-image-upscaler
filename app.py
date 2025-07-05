import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Check for GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)

# Setup the upsampler
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,  # Use False for CPU
    device=device
)

# Image processing function
def upscale_image(image, factor):
    try:
        if image is None:
            return "⚠️ No image received. Please upload or recapture."

        # Resize large images to speed up processing
        max_dim = 512
        width, height = image.size
        if max(width, height) > max_dim:
            ratio = max_dim / float(max(width, height))
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height))

        # Convert to OpenCV format
        img = np.array(image)
        if img.ndim != 3 or img.shape[2] != 3:
            return "⚠️ Invalid image format. Please use a proper color image."

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Upscale using RealESRGAN
        output, _ = upsampler.enhance(img, outscale=int(factor))
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output)

    except Exception as e:
        return f"❌ Error during upscaling: {str(e)}"

# Gradio UI
demo = gr.Interface(
    fn=upscale_image,
    inputs=[
        gr.Image(type="pil", label="Upload or Webcam"),
        gr.Radio(choices=["2", "4"], value="4", label="Upscale Factor")
    ],
    outputs=gr.Image(type="pil", label="Upscaled Output"),
    title="🖼️ AI Image Upscaler (Real-ESRGAN)",
    description="Upload or capture image via webcam. Choose upscale factor (2x or 4x) and get a high-resolution output."
)

demo.launch()
