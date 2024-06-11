import torch
import torch.nn.functional as F
from accelerate import Accelerator
from mri3d.models_onlydec.vae_std_zeroin import AutoencoderKL3D
# from mri3d.data.dataset_mri import MRIDataset
from einops import rearrange
from dataloader import get_data_loaders
from empatches import EMPatches
import matplotlib.pyplot as plt
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np
import math
import os
import nibabel as nib
#hypara##########################################
learning_rate = 1e-5
trainable_modules: Tuple[str] = (
    "conv11",
    "conv21",
    "temp",
)
weight_dtype = torch.float16
model_path = './checkpoints'
enable_xformers_memory_efficient_attention:bool = True
scale_lr: bool = False
lr_scheduler: str = "constant"
lr_warmup_steps: int = 0
adam_beta1: float = 0.9
adam_beta2: float = 0.999
adam_weight_decay: float = 1e-4
adam_epsilon: float = 1e-08
max_grad_norm: float = 1.0
gradient_accumulation_steps: int = 1
gradient_checkpointing: bool = True
checkpointing_steps: int = 500
max_train_steps: int = 30*12*4000
mixed_precision: Optional[str] = "fp16"
accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision, )
##################################################
emp = EMPatches()
#reshape
def extract_patches_5d(source, patchsize=(144,120,120), overlap=0.25, stride=None, emp=emp, vox=True):
    a= []
    assert len(source.shape) == 5
    for i in range(source.shape[1]):
        patches, indices = emp.extract_patches(source[0,i,:,:,:],patchsize=patchsize, overlap=overlap, stride=stride, vox=vox)
        a.append(torch.stack(patches).unsqueeze(1).unsqueeze(1))
        # print(torch.stack(patches).unsqueeze(1).unsqueeze(1).shape)
    a = torch.cat(a, dim=2)
    return a

checkpoint_path = "./recon_newvae_deczero_whole/checkpoints/vaenew_model_epoch_2_source_1499.pth"
#load model
vae = AutoencoderKL3D.from_pretrained_2d(pretrained_model_path=model_path, subfolder='vae', strict=False)
vae.load_state_dict(torch.load(checkpoint_path))

vae.requires_grad_(True)
for name, module in vae.named_modules():
    if name.endswith(tuple(trainable_modules)):
        for params in module.parameters():
            params.requires_grad = False


if gradient_checkpointing:
        vae.enable_gradient_checkpointing()

if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
#load optimizer
optimizer = torch.optim.AdamW(vae.parameters(),lr=learning_rate,
                             betas=(adam_beta1, adam_beta2),
                             weight_decay=adam_weight_decay,
                             eps=adam_epsilon,)
#load data
train_mri, val_mri = get_data_loaders("MRI", 176, 1)

# Scheduler
lr_scheduler = get_scheduler(
    lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
    num_training_steps=max_train_steps * gradient_accumulation_steps,)


#prepare everything
vae, optimizer, train_data, lr_scheduler = accelerator.prepare(vae, 
    optimizer, train_mri, lr_scheduler)


# train
vae.to(accelerator.device)

# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_data)*12/ gradient_accumulation_steps)
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

losses = []

# def visualize_reconstructions(input_patches, output_dir, indices, epoch, source_idx, emp):
#     merge_p = emp.merge_patches(input_patches[:,:,0,:,:,:], indices, mode='avg')
#     plt.imshow(merge_p[90,:,:], cmap='gray')
#     plt.set_title('Original')
#     plt.savefig(f'{output_dir}/epoch_{epoch}_source_{source_idx}_patch_{i}.png')
#     plt.close()

def save_png(recimg, recon_dir, slice_num, epoch, source_idx):
    plt.imshow(recimg[0,0,:,slice_num,:],cmap='gray')
    plt.savefig(recon_dir+f'epoch_{epoch}_source1_{source_idx}.png')
    plt.close()
    plt.imshow(recimg[0,0,slice_num,:,:],cmap='gray')
    plt.savefig(recon_dir+f'epoch_{epoch}_source2_{source_idx}.png')
    plt.close()
    plt.imshow(recimg[0,0,:,:,slice_num],cmap='gray')
    plt.savefig(recon_dir+f'epoch_{epoch}_source3_{source_idx}.png')
    plt.close()

def save_nii_image(nii_array,path):
    # nii_array = nii_input.cpu().detach().numpy()
    nii_array = nii_array.mean(axis=1).squeeze()
    nii_image = nib.Nifti1Image(nii_array, affine=np.eye(4))
    nib.save(nii_image, path)

slice_num = 88

def bulidfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
out_dir = 'recon_newvae_deczero_whole_newdata/'
bulidfolder(out_dir)
recon_dir = out_dir + 'img/recon/'
origin_dir = out_dir + 'img/origin/'
nii_dir = out_dir + 'img/nii/'
check_dir = out_dir + 'checkpoints/'
bulidfolder(recon_dir)
bulidfolder(check_dir)
bulidfolder(origin_dir)
bulidfolder(nii_dir)


beta = 0
for epoch in range(num_train_epochs):
    vae.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_data, desc=f'Epoch {epoch+1}/{num_train_epochs}', leave=False)
    for source_idx, source in enumerate(progress_bar):
        with accelerator.accumulate(vae):
            patches = extract_patches_5d(source, patchsize=(144,88,88), overlap=0.25, stride=None, vox=True)
            patches = patches.to(weight_dtype)
            # output = torch.zeros_like(patches)
            for i in range(patches.shape[0]):
                input = rearrange(patches[i], 'b c f h w -> (b f) c h w')
                output = vae(input).sample
                # kl_loss = vae.encode(patches[i].float()).latent_dist.kl()
                loss = F.mse_loss(output.float(), patches[i].float())
                # loss = recon_loss + beta * kl_loss
                accelerator.backward(loss)
                optimizer.step()
                epoch_loss += loss.item()
                lr_scheduler.step()
                optimizer.zero_grad()
                # patches[i] = output
                print(f"Patch [{i+1}/{patches.shape[0]}], Loss: {loss.item()}")
            progress_bar.set_postfix({'total_loss':loss.item()})
            if (source_idx+1) % 1400 == 0:
                # visualize_reconstructions(output, recon_dir, indices, epoch, source_idx, emp)
                torch.save(vae.state_dict(), os.path.join(check_dir, f'vaenew_model_epoch_{epoch}_source_{source_idx}.pth'))
                with torch.no_grad():
                    print(source.shape)
                    f = source.shape[2]
                    source = rearrange(source, 'b c f h w -> (b f) c h w')
                    recimg = vae(source)
                    source = rearrange(source, ' (b f) c h w ->b c f h w ',f=f)
                    recimg = recimg.sample.cpu().detach().numpy()
                    print(recimg.shape)
                    source = source.cpu().detach().numpy()
                    #origin
                    save_png(source,origin_dir,slice_num,epoch,source_idx)
                    save_png(recimg,recon_dir,slice_num,epoch,source_idx)
                    save_nii_image(recimg, nii_dir+f'{epoch}_{source_idx}.nii')
    epoch_loss /= len(train_data)
    losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_train_epochs}], Loss: {epoch_loss}")
    torch.save(vae.state_dict(), os.path.join(check_dir, f'vae2_model_epoch_{epoch}.pth'))

# torch.save(vae.state_dict(), 'vae_model.pth')


# plt.plot(losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.show()
