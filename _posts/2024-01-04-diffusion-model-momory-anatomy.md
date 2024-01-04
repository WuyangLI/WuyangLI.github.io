# Avatar Diffusion Model Memory Anatomy
Following the blog ["build-avatar-diffusion-model-from-scratch"](https://wuyangli.github.io/2024/01/01/build-avatar-diffusion-model-from-scratch), 

During interations with the model, encountering an "OutOfMemory" error is not uncommon. It can be puzzling to see that a relatively small model consumes the entire GPU memory.

Trainable parameters of the avatar diffusion model is 76,464,439 (76 million), thus it's of similar size to Resnet152, which has 60 million parameters.

## Components on GPU memory
As outlined in [the huggingface blog -model memory anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy), the components on GPU memory are the following: 
1. model weights
2. optimizer states
3. gradients
4. forward activations saved for gradient computation
5. temporary buffers
6. functionality-specific memory

As shown in the screenshot of `nvidia-smi` on an AWS g5 2xlarge instance with 24G GPU memory, the training of the model consumes up to 22616 MB GPU memory. Among the components listed above, which is the top culprit? 
<p align=center>
  <img src="/docs/assets/images/diffusion_models/figures/nvidia-smi-screenshot.png" alt="" width="600"/>
</p>

## The Culprit
### Model Weights, Optimizer and Gradients
the following table is a breakdown of model memory of weights, optimizer and gradients:
|memory breakdown|bytes per parameter|memory|
|:---------------- | :------: | ----: |
|model weight|4|291.69 MB|
|adam optimizer|8|583.38 MB|
|gradients|4|291.69 MB|

As you can see, they're nothing compared to the total GPU consumption. 
### Forward Activations
How about forward activations that are saved for gradient computation? I used the following [function](https://discuss.pytorch.org/t/pytorch-appears-to-be-crashing-due-to-oom-prematurely/131039/13) to estimate the activatio size of the avatar diffusion model:
```python
# modified from https://discuss.pytorch.org/t/pytorch-appears-to-be-crashing-due-to-oom-prematurely/131039/13
# credit to ndvbd
total_output_elements = 0
def calc_total_activation_size(model, input_tensor_size):
    """
    given the avatar diffusion model, calculate the activation memory in MB 
    """
    global total_output_elements
    total_output_elements = 0

    def hook(module, input, output):
        global total_output_elements
        if torch.is_tensor(output):
            total_output_elements += output.numel()
        # tuple output for attention modules
        elif isinstance(output, tuple):
            for o in output:
                if torch.is_tensor(o):
                    total_output_elements += o.numel()
        
    handle = torch.nn.modules.module.register_module_forward_hook(hook)
    x = torch.randn(*input_tensor_size)
    t_test = torch.randint(1, 10, (x.shape[0],)).float()
    context_mask = torch.bernoulli(torch.zeros_like(c)+0.2)
    result = model(x, c, t_test, context_mask)
    handle.remove()
    return total_output_elements*4/(1024*1024)

calc_total_activation_size(nn_model, (128, 3, 64, 64))
```

