---
layout: post
title: "Build Cartoon Avatar Diffusion Model with HuggingFace diffusers [WIP]"
date: 2024-01-11
author: Wuyang
categories: diffusion model
permalink: build-cartoon-avatar-diffusion-model-with-hg-diffusers
---
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_cartoon_set_diffusion_inference_gs_7_sampled.gif" alt="cartoon avatar diffusion random samples hg diffusers" width="500"/>
</p>

[notebook](diffusion_models/cartoonset_diffusion/diffuser_cartoonset_diffusion_conditional.ipynb)

## Unet built with HG diffusers
```python
UNet2DConditionModel((64, 64), 3, 3, 
                            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D","CrossAttnUpBlock2D"),
                            block_out_channels=(128, 256, 512),
                            cross_attention_dim=1280,
                            layers_per_block=2,
                            only_cross_attention=True,
                            attention_head_dim=32)

```
```
trainable model parameters: 188511363
all model parameters: 188511363
percentage of trainable model parameters: 100.00%
model weight size: 719.11 MB
adam optimizer size: 1438.23 MB
gradients size: 719.11 MB
```
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_6.png" alt="attn head 32 cross attn only" width="500"/>
</p>

### Other Attempts
```python
UNet2DConditionModel((64, 64), 3, 3, 
                            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D","CrossAttnUpBlock2D"),
                            block_out_channels=(128, 256, 512),
                            cross_attention_dim=1280,
                            layers_per_block=2,
                            attention_head_dim=16)

```
```
trainable model parameters: 180483203
all model parameters: 180483203
percentage of trainable model parameters: 100.00%
model weight size: 688.49 MB
adam optimizer size: 1376.98 MB
gradients size: 688.49 MB
```

<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_attn_head_16_epoch_6.png" alt="attn head 16" width="500"/>
</p>

```python
UNet2DConditionModel((64, 64), 3, 3, 
                            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D","CrossAttnUpBlock2D"),
                            block_out_channels=(128, 256, 512),
                            cross_attention_dim=1280,
                            layers_per_block=2,
                            attention_head_dim=8) # attention_head_dim default value is 8

```
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_atten_head_8_epoch_15.png" alt="attn head 8" width="500"/>
</p>

## Play with guidance scale

<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_cartoon_set_diffusion_guidance_scale_from_0.0_9.0.gif" alt="cartoon avatar diffusion guidance scale from 0 to 9" width="400"/>
</p>

when guidance is 0

<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_0.0.png" alt="cartoon avatar diffusion guidance 0" width="400"/>
</p>

when guidance is 1.0
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_1.0.png" alt="cartoon avatar diffusion guidance 1.0" width="400"/>
</p>

when guidance is 2.0
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_2.0.png" alt="cartoon avatar diffusion guidance 2.0" width="400"/>
</p>

when guidance is 9.0
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_9.0.png" alt="cartoon avatar diffusion guidance 9.0" width="400"/>
</p>

## What does the model learn over the epochs?

<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_0.png" alt="random samples of cartoon avatar diffusion epoch 0" width="400"/>
</p>
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_1.png" alt="random samples of cartoon avatar diffusion epoch 1" width="400"/>
</p>
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_2.png" alt="random samples of cartoon avatar diffusion epoch 2" width="400"/>
</p>
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_3.png" alt="random samples of cartoon avatar diffusion epoch 3" width="400"/>
</p>

## Model built with HG diffusers vs from scratch
**Not an apple-to-apple comparison**

||model built from scratch|model built with diffusers|
|:---|:---|:---|
|number of parameters|76M|180M|
|denoising model|Tau + Unet|Tau + Unet|
|Unet basic blocks|resnet blocks, multi-head cross attention|resnet blocks, transformer block (cross attention only)|
|number of attention head|4|32|
|training time per epoch|~10 mins|~120 mins|
|optimizer epoch|Adam|AdamW|
|learning rate schedule|StepLR|cosine_schedule_with_warmup|
|strength|accurate shape, various hairstyles|clean background, bright color|
|weakness|noisy, color corruption| hairstyle not well captured|
|example|<img align=center src="/docs/assets/images/diffusion_models/figures/scratch_gs_2_attn_head_4.png" alt="" width="64"/>|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_gs_2_attn_head_32.png" alt="" width="64"/>|


