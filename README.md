
# Automatic Image Colorization with U-Net and Perceptual Loss

This project features an automatic image colorization model that leverages a U-Net architecture combined with perceptual loss for enhanced colorization quality. The U-Net captures intricate image features through its encoder-decoder structure, while perceptual loss, utilizing a pre-trained VGG network, ensures the model generates visually pleasing and realistic colors by comparing high-level features between the generated and true color images. This approach balances pixel-level accuracy and perceptual realism, leading to superior colorization results. The project includes training scripts and sample results, providing a comprehensive framework for automatic image colorization.

## Table of Contents

	1.	Introduction
	2.	Model Architecture
	3.	Perceptual Loss
	4.	Installation
	5.	Usage

### Introduction

Image colorization is a challenging and fascinating problem in computer vision. This project presents a solution using a U-Net architecture combined with perceptual loss, aiming to generate high-quality, realistic color images from grayscale inputs.

### Model Architecture

The core of the colorization model is the U-Net architecture, which consists of an encoder-decoder structure. The encoder compresses the input image into a lower-dimensional feature space, capturing essential features, while the decoder reconstructs the image from this compressed representation, adding color in the process.

### Perceptual Loss

Perceptual loss is employed to enhance the realism of the colorized images. It uses a pre-trained VGG network to compare high-level features between the generated and true color images. This ensures that the colorization process not only focuses on pixel-level accuracy but also maintains perceptual quality.

### Installation

To set up the project, follow these steps:

Clone the repository:
```bash
git clone https://github.com/Ayush-Singh677/Image-Colorisation.git
cd Image-Colorisation
```

Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required dependencies:
```
pip install -r requirements.txt
```

### Usage
```
python3 main.py --input grayscale_image.jpg --output colorized_image.jpg --model model.pth
```

