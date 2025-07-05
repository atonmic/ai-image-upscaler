# ai-image-upscaler
# 🖼️ AI Image Upscaler (Real-ESRGAN)

This project is a web app that uses [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to upscale low-resolution images into high-resolution, sharper outputs using deep learning.

## 🚀 Features

- 📤 Upload or use webcam to capture image
- 🔍 Choose 2x or 4x upscale
- 💡 Real-time image enhancement using Real-ESRGAN
- 🖥️ Gradio UI for seamless interaction

## 🔧 Tech Stack

- Python
- PyTorch
- Real-ESRGAN
- OpenCV
- Gradio
- Hugging Face Spaces

## 🧠 Model Used

- `RealESRGAN_x4plus.pth` (General purpose upscaling model)

## 📦 Setup Instructions

```bash
git clone https://github.com/YOUR_USERNAME/ai-image-upscaler.git
cd ai-image-upscaler
pip install -r requirements.txt
python app.py

Built an AI-powered image upscaler using Real-ESRGAN to convert low-resolution images into high-quality outputs with 2x and 4x enhancement options.
 This project is a deep learning-based AI image upscaler that enhances low-resolution images into sharp, high-resolution outputs using the Real-ESRGAN model. Built using Python, PyTorch, OpenCV, and Gradio, the app allows users to upload or capture images via webcam, choose their desired upscale factor (2x or 4x), and view or download the enhanced output — all from a simple web interface.

✅ Features:

Upload or use webcam to input image

Select 2x or 4x upscale factor

Real-time enhancement with Real-ESRGAN

Clean Gradio-based UI

Deployed live on Hugging Face Spaces

This project showcases practical use of super-resolution models in a real-world web app, optimized for CPU environments with a focus on speed, stability, and ease of use.
