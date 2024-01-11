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


## Play with guidance scale

<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_cartoon_set_diffusion_guidance_scale_from_0.0_9.0.gif" alt="cartoon avatar diffusion guidance scale from 0 to 9" width="400"/>
</p>
