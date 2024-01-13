---
layout: post
title: "Build Cartoon Avatar Diffusion Model using HuggingFace diffusers"
date: 2024-01-11
author: Wuyang
categories: diffusion model
permalink: build-cartoon-avatar-diffusion-model-using-hg-diffusers
---
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_cartoon_set_diffusion_inference_gs_7_sampled.gif" alt="cartoon avatar diffusion random samples hg diffusers" width="500"/>
</p>

Continuing from the last project ["build cartoon avatar diffusion model from scratch"](https://wuyangli.github.io/build-avatar-diffusion-model-from-scratch), I crafted a new model for generating cartoon avatars using the Huggingface Diffusers library.

For additional insights into this model, refer to the linked Jupyter note [notebook](diffusion_models/cartoonset_diffusion/diffuser_cartoonset_diffusion_conditional.ipynb) for comprehensive details.

The project setup closely mirrors the previous one, encompassing the dataset, condition component, noise schedule, and denoising process, which all remain identical. The divergences lie in two key aspects:
1. Unet (implementation and structure)
2. Optimizer and learning rate schedule


## Unet built with HG diffusers
As shown in the following code block, the UNet is an instance of `UNet2DConditionModel` in the Huggingface diffusers library. Both `down` and `up` blocks are `CrossAttn` blocks. What stands out is that the model utilizes 32 attention heads.
Apparently, the Unet is more complex in terms of structure than the one we built from scratch. Does increased model complexity result in visually superior generated images?
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
Below are a random samples of generated cartoon avatars after training for 6 epochs.
<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_6.png" alt="attn head 32 cross attn only" width="800"/>
</p>
<p align=center>generated images at epoch 6 </p>

How do they compare to the images generated in the project ["build cartoon avatar diffusion model from scratch"](https://wuyangli.github.io/build-avatar-diffusion-model-from-scratch) ?

<p align=center>
  <img src="/docs/assets/images/diffusion_models/figures/figure16_conditional_diffusion_4_ep18.png" alt="figure 16" width="800"/>
</p>
<p align=center>generated images of the model built from scratch at epoch 18</p>

### Other Attempts
I also trained Unets with other settings, their main differences from the model outline above are:
1. They use both cross-attention and self-attention.
2. They use fewer attention heads, 16 and 8 respectively. whereas the final version of the model uses 32 attention heads.

Observing the generated images gives us a clear indication:
With 8 attention heads, the model struggles to accurately depict color and hairstyle.

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
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_atten_head_8_epoch_15.png" alt="attn head 8" width="800"/>
</p>

Increasing the number of heads to 16 improves color representation, but the portrayal of hairstyles remains lacking.

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
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_attn_head_16_epoch_6.png" alt="attn head 16" width="800"/>
</p>


## Play with guidance scale

We used a fixed guidance 2.0 in the last blog and didn't play with this parameter.
According to [Exploring stable diffusion guidance](https://10maurycy10.github.io/misc/sd_guidance/), 
>The guidance scale parameter controls how strongly the generation process is guided towards the prompt. A value of 0 is equivalent to no guidance, and the image is completely unrelated to the prompt (it can however, still be an interesting image). Increasing the guidance scale increases how closely the image resembles the prompt, but reduces consistency and tends to lead to an overstaturated and oversharpened image.

Can we witness similar influences of guidance on the cartoon avatar model? 

Using identical condition embeddings, I ramped up the guidance from 0.0 to 9.0 and produced 16 samples for each guidance value. 

The table below illustrates that at a guidance value of 0, images appear random. By the time the guidance reaches 1.0, the images already show discernible patterns, and there isn't a significant qualitative distinction among images generated with guidance values ranging from 2.0 to 9.0.

<p align=center>
  <img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_cartoon_set_diffusion_guidance_scale_from_0.0_9.0.gif" alt="cartoon avatar diffusion guidance scale from 0 to 9" width="800"/>
</p>

|guidance scale|generated images given the same conditions|
|:----:|:----:|
|0.0|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_0.0.png" alt="cartoon avatar diffusion guidance 0" width="800"/>|
|1.0|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_1.0.png" alt="cartoon avatar diffusion guidance 1.0" width="800"/>|
|2.0|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_2.0.png" alt="cartoon avatar diffusion guidance 2.0" width="800"/>|
|3.0|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_3.0.png" alt="cartoon avatar diffusion guidance 2.0" width="800"/>|
|5.0|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_5.0.png" alt="cartoon avatar diffusion guidance 2.0" width="800"/>|
|7.0|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_7.0.png" alt="cartoon avatar diffusion guidance 2.0" width="800"/>|
|9.0|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_inference_image_gs_9.0.png" alt="cartoon avatar diffusion guidance 9.0" width="800"/>|

## What does the model learn over the epochs?

|epoch|random samples|
|:---:|:---:|
|0|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_0.png" alt="random samples of cartoon avatar diffusion epoch 0" width="800"/>|
|1|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_1.png" alt="random samples of cartoon avatar diffusion epoch 1" width="800"/>|
|2|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_2.png" alt="random samples of cartoon avatar diffusion epoch 2" width="800"/>|
|3|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_image_epoch_3.png" alt="random samples of cartoon avatar diffusion epoch 3" width="800"/>|


after epoch 3, the loss doesn't decrease dramatically and gradually statuates at epoch 6.

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
|strength|excels in accurately depicting shapes, particularly a diverse range of hairstyles|produces images with pristine backgrounds and vibrant colors|
|weakness|tends to generate images with numerous noisy pixels and occasional color corruption|struggles in accurately portraying various styles of hairstyles|
|example|<img align=center src="/docs/assets/images/diffusion_models/figures/scratch_gs_2_attn_head_4.png" alt="" width="64"/>|<img align=center src="/docs/assets/images/diffusion_models/figures/hg_diffusers_gs_2_attn_head_32.png" alt="" width="64"/>|


