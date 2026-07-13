# 🎨 Perspecta AI

<p align="center">

<img src="assets/banner.png" width="100%">

</p>

<p align="center">

Real-Time Arbitrary Neural Style Transfer using <b>Adaptive Instance Normalization (AdaIN)</b> and a <b>VGG-19 Encoder</b>.

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![Deep Learning](https://img.shields.io/badge/DeepLearning-NeuralStyleTransfer-success)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# 📖 Overview

**Perspecta AI** is a real-time neural style transfer application that transforms an ordinary photograph into an artistic image while preserving the original scene.

Unlike traditional neural style transfer methods that require optimization for every single image, Perspecta AI performs **style transfer in one forward pass**, making it significantly faster and suitable for real-time applications.

The project is built using:

- PyTorch
- Flask
- Adaptive Instance Normalization (AdaIN)
- VGG-19 Encoder
- Custom-trained Decoder Network

---

# ✨ Features

- 🎨 Arbitrary Neural Style Transfer
- ⚡ Real-Time Image Generation
- 🧠 Adaptive Instance Normalization (AdaIN)
- 🏗️ Pretrained VGG-19 Feature Extractor
- 🚀 Custom Decoder trained from scratch
- 🌐 Flask Web Interface
- 📤 Upload Content & Style Images
- 🎚 Adjustable Style Strength (Alpha)
- 💻 Apple MPS / CUDA / CPU Support
- 🧩 Clean Modular Codebase

---

# 🧠 What is Neural Style Transfer?

Neural Style Transfer is a Deep Learning technique that combines:

- **Content** from one image
- **Style** from another image

to generate a completely new artistic image.

Example

```
Content Image
       +
Style Image
       ↓
Stylized Image
```

Unlike traditional image filters, Neural Style Transfer understands high-level image representations learned by convolutional neural networks.

---

# 🚀 Why AdaIN?

Earlier Style Transfer methods required:

- hundreds of optimization iterations
- expensive computation
- one network per style

Adaptive Instance Normalization solves these limitations.

AdaIN directly aligns the statistics of the content features with the statistics of the style features.

Instead of optimizing pixels, AdaIN modifies deep feature representations.

This enables:

- Arbitrary Style Transfer
- One model for infinite styles
- Real-Time inference

---

# 📐 Adaptive Instance Normalization

Given

Content Features

```
Fc
```

Style Features

```
Fs
```

AdaIN computes

```
AdaIN(Fc, Fs)

=

σ(Fs)

×

(Fc − μ(Fc))

──────────────

σ(Fc)

+

μ(Fs)
```

where

- μ = Mean
- σ = Standard Deviation

The content feature is first normalized and then re-scaled using the style feature statistics.

This transfers the visual style while preserving semantic content.

---

# 🏗 Why VGG-19?

VGG-19 is used as a **fixed feature extractor**.

Instead of using raw pixels, style transfer is performed on deep feature maps.

The encoder captures:

- edges
- textures
- shapes
- semantic information

Only the encoder is frozen.

The decoder is trained to reconstruct an RGB image from transformed feature maps.

---

# 🏛 Project Architecture

```text
                 Style Image
                      │
                      ▼
                VGG Encoder
                      │
               Style Features
                      │
                      │
Content Image         │
      │               │
      ▼               │
 VGG Encoder          │
      │               │
 Content Features     │
      │               │
      └──────┬────────┘
             ▼
           AdaIN
             │
             ▼
        Stylized Features
             │
             ▼
          Decoder
             │
             ▼
      Stylized Image
```

---

# 🔄 Training Pipeline

```text
Content Image
      │
      ▼
 VGG Encoder
      │
      ▼
Content Features

Style Image
      │
      ▼
 VGG Encoder
      │
      ▼
Style Features

      │
      ▼

 Adaptive Instance Normalization

      │
      ▼

 Decoder

      │
      ▼

 Generated Image

      │
      ▼

 VGG Encoder

      │
      ▼

Content Loss

Style Loss

      │
      ▼

 Backpropagation
```

---

# ⚙️ Inference Pipeline

```text
Content Image

      │

      ▼

VGG Encoder

      │

      ▼

Content Features

               Style Image

                    │

                    ▼

              VGG Encoder

                    │

                    ▼

              Style Features

                    │

                    ▼

                 AdaIN Layer

                    │

                    ▼

                 Decoder

                    │

                    ▼

             Stylized Image
```

---

# 📊 Loss Function

## Content Loss

The generated image should preserve the semantic information of the original image.

Content Loss is computed as

```
MSE(Generated Features,
    AdaIN Features)
```

---

## Style Loss

Instead of Gram Matrices, AdaIN matches

- Mean
- Standard Deviation

of feature maps.

For each encoder layer

```
Loss

=

MSE(μg, μs)

+

MSE(σg, σs)
```

where

- μg = Generated Mean
- σg = Generated Std
- μs = Style Mean
- σs = Style Std

---

# 🧩 Folder Structure

```
PerspectaAI/

│

├── app.py

├── train.py

├── requirements.txt

├── README.md

│

├── utils/

│ ├── models.py

│ └── utils.py

│

├── templates/

│ └── index.html

│

├── static/

│ ├── uploads/

│ ├── css/

│ └── js/

│

├── examples/

│

└── experiment/
```

---

# 📚 Dataset

The decoder was trained using

- Large Content Dataset
- Large Style Dataset

Image Size

```
512 × 512
```

Random Crop

```
256 × 256
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/PerspectaAI.git

cd PerspectaAI
```

Install dependencies

```bash
uv sync
```

or

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

```bash
python train.py \
--batch_size 4 \
--epochs 60 \
--experiment big_dataset
```

---

# 🌐 Run Web Application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 🎚 Adjustable Style Strength

The application allows users to control the intensity of style transfer using the Alpha parameter.

```
Alpha = 0

↓

Original Content

Alpha = 0.5

↓

Balanced Style

Alpha = 1

↓

Maximum Style
```

---

# 💻 Tech Stack

- Python
- PyTorch
- TorchVision
- Flask
- HTML
- CSS
- JavaScript
- Bootstrap
- PIL
- tqdm

---

# 🚀 Future Improvements

- Video Style Transfer
- Style Interpolation
- ONNX Export
- TensorRT Optimization
- Docker Deployment
- Cloud Deployment
- User Gallery
- Batch Processing
- Style Recommendation
- Mobile Application

---

# 📈 Results

| Content | Style | Output |
|----------|---------|---------|
| *(Add Image)* | *(Add Image)* | *(Add Image)* |

---

# 📖 References

1. Huang, X., & Belongie, S. (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.*

2. Simonyan, K., & Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition.*

3. PyTorch Documentation

4. Flask Documentation

---

# 👨‍💻 Author

**Hemant Machiwar**

B.Tech Mathematics & Computing  
National Institute of Technology Hamirpur

GitHub: https://github.com/yourusername

---

## ⭐ If you found this project useful, consider giving it a Star!
