# 3D MRI Brain Image Generation Using Latent Diffusion Models

This repository is a Stanford CS231N project. It contains the implementation of our project on generating realistic 3D MRI brain images by adapting a 2D latent diffusion model to operate in a three-dimensional framework. Our approach focuses on architectural modifications to effectively process 3D data, demonstrating superior performance over traditional GAN-based methods and a much faster sampling time than SOTA diffusion-based methods.

## Project Overview

We have extended the stable diffusion model architecture to generate synthetic 3D MRI images that maintain structural integrity and temporal consistency. This adaptation is beneficial for medical research and educational purposes, providing a scalable solution to data scarcity in medical imaging.

## Features

- **Temporal-Consistent VAE**: Modifications include the integration of 1D convolution layers to capture temporal dynamics.
- **Spatial-Temporal UNet**: Enhancements involve adding temporal blocks within the UNet architecture to process 3D data efficiently.


#### to train temporal-consistent vae, run
```bash
accelerate launch --mix_precision fp16 train_vae_deczero.py
```

#### to train temporal-consistent unet, run
```bash
accelerate launch --mix_precision fp16 train_onlydec_3d.py
```
