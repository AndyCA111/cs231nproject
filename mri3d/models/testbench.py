from vae import AutoencoderKL3D
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
import numpy as np
import torch
from torch import nn
import sys
sys.path.append('../mri3d')


weight_dtype = torch.float16
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = '/scratch/project_2002846/Binxu/3dbrain_generation/checkpoints'
vae = AutoencoderKL3D.from_pretrained_2d(pretrained_model_path=model_path, subfolder='vae', strict=False)
# vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae')
# vae.requires_grad_(False)
vae = vae.to(device, dtype=weight_dtype)
a = torch.rand(1,3,96,96,96)
# a = a.float()
a = a.to(device, dtype=weight_dtype)

out = vae(a)
out = out.sample.cpu().detach().numpy()
print(vae)
print(out.shape)