# cs231n project

### in this project we extend 2D pretrained stable diffusion to do 3D MRI generation


#### to train temporal-consistent vae, run
```bash
accelerate launch --mix_precision fp16 train_vae_deczero.py
```

#### to train temporal-consistent unet, run
```bash
accelerate launch --mix_precision fp16 train_onlydec_3d.py
```