---
title: 'Diffusers'
date: 2024-02-15
permalink: /posts/24/02/diffusers
tags:
  - Diffusion
use_math: true
---

破碎的梦的开始。

Diffusers 是 Diffusion 模型的代码框架包装，其最大的单位是 pipeline。你只需要一个 pipeline 和权重文件（夹），就可以做到几行代码生成图片。例子如下：
```python
from diffusers import DiffusionPipeline
import torch
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5", torch_dtype=torch.float16, variant='fp16')
pipe.to("cuda")
pipe("A dog").images[0]
```
现在一些论文（比如 layout-guidance，简称 tflcg）利用 diffusers 库，结合自己的算法提供了一个 pipeline 供大家使用，比如其库提供的示例代码：
```python
from tflcg.layout_guidance_pipeline import LayoutGuidanceStableDiffusionPipeline
from diffusers import EulerDiscreteScheduler
import transformers
import torch
transformers.utils.move_cache()
pipe = LayoutGuidanceStableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5", torch_dtype=torch.float16, variant='fp16')
pipe = pipe.to("mps")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
prompt = "A cat playing with a ball"
bboxes = [[0.55, 0.4, 0.95, 0.8]]
image = pipe(prompt, num_inference_steps=20,
             token_indices=[[2]],
             bboxes=bboxes).images[0]
image = pipe.draw_box(image, bboxes)
image.save("output.png")
```
为了能够在未来的工作中写出类似的代码，我觉得我有必要研究一下 diffusers 库的相关代码细节。我参考的是知乎上大佬的[这篇文章](https://zhuanlan.zhihu.com/p/672574978)并且补充了一些细节。

## Pipeline

一个 pipeline 包含了：

- VAE、UNet、Text Encoder 三个大模块。

- Tokenizer、Scheduler、Safety checker、Feature extractor 四个小模块。

其中后两个是用来检测是否健康的（你懂的），也称作 NSFW 检测器（Not Safe For Work）。我们使用 from_pretrained 函数读取一个 pipeline，这个函数会根据文件夹中的 `model_index.json` 新建一个 pipeline 类。比如 SD 1.5 的 `model_index.json` 文件如下：

```json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.6.0",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```
`_class_name` 告诉你要变成啥 pipeline，比如 SD 就是 StableDiffusionPipeline。然后后面就是从哪个库里读取什么类作为 pipeline 这个模块的类，比如 VAE 模块对应的是 diffusers 库中的 AutoencoderKL 类。

## StableDiffusionPipeline

我们来看 StableDiffusionPipeline 的 `__init__`：

```python
class StableDiffusionPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin
): # 继承了一堆东西，比如最基本的 Diffusion Pipeline
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        如果 *** 缺少信息，给出警告。
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) # VAE 的压缩率?
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
```

> 这个变量跟着一个冒号的叫做类型注解，用于注释，不影响代码运行（也就是说，即使调用函数时形参类型不匹配，python 也不会报错）

然后一堆 hasattr 函数检查 config 文件有没有漏信息，给出警告。

最后这个 register_modules 是 Diffusion Pipeline 内置的函数：

```python
    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}
            # save model index config
            self.register_to_config(**register_dict)
            # set models
            setattr(self, name, module)
```

不讲废话，然后是 `__call__` 部分，也就是生成图片的部分。

```python
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
```

这里有一堆参数，看过论文的话，字面意思大多都能理解（除了 negative prompt 部分我也没看过，还有 callback 这一些，先不管）

然后做了一个输入参数的 check，点进去特别好玩，有一堆 raise Error。

```python
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
```

这段代码把 batch size 设为 prompt 的个数，并行计算。


```python
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
```

进行编码。

```python
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
```

设置时间参数。这种方法有很多，比如我们看 tflcg 示例代码里：

```python
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
```

就可以发现，他用的是 EulerDiscreteScheduler ~~，虽然我也不知道是啥~~。SD 1.5 默认的话是 PNDMScheduler。 ~~我还是不知道是啥。~~

```python
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
```

初始噪声的确定，返回 [batch size, 4, 64, 64] 的 tensor。它里面包装的很好，generator 甚至可以是一个 list。

后面的整个循环就是经典 DDPM/DDIM 采样。由于我没看过 classifier free guidance，所以没仔细看，直接咕咕咕了。

```python
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
```

#### function: scale_model_input

`scheduler.scale_model_input` 是什么意思？我翻了源码，比如 `scheduling_euler_discrete.py` 是这样的：

```python
    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        self.is_scale_input_called = True
        return sample
```

我猜测是根据不同的 scheduler 对不同的输入进行预处理，事实证明应该就是这样，因为我翻了 `scheduling_ddim.py`，发现了一个很搞笑的：

```python
    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        return sample
```

处理就是：**啥都不处理**。

这些递推函数什么的，都在 scheduler 里封装好了。实在是太酷了。

## LayoutGuidanceStableDiffusionPipeline

可以发现，它的 `__call__` 几乎就是在 SD 的基础上魔改的。只不过它多传了 `token_indices` 和 `bboxes`。我们调不一样的地方看。

首先它继承了 `StableDiffusionAttendAndExcitePipeline`，也就是说，它其实是在 AAE 的基础上做的，为啥论文里都没写。

> 值得一提的是，我看的 diffusers 是最新版，但是 tflcg 基于的是 0.15.0，所以一些细节会不一样，支持的功能可能也不一样。

> 另外值得一提的是，AAE 已经被 diffusers 删了，成为时代的眼泪了（难绷，也就一年啊）。

看关键的反向传播部分：

```python
     with torch.enable_grad():
         latents = latents.clone().detach().requires_grad_(True) # 先克隆，单独拿出来，为了不影响计算图
         for guidance_iter in range(max_guidance_iter_per_step): # 反向传播几次，论文也写了，默认 5 次，
             if loss.item() / scale_factor < 0.2:
                 break
             latent_model_input = self.scheduler.scale_model_input(
                 latents, t
             )
             self.unet(                                          # 过一遍 Unet
                 latent_model_input,
                 t,
                 encoder_hidden_states=cond_prompt_embeds,
                 cross_attention_kwargs=cross_attention_kwargs,
             )
             self.unet.zero_grad()
             loss = (                                            # 根据论文描述的过 loss
                 self._compute_loss(token_indices, bboxes, device)
                 * scale_factor
             )
             grad_cond = torch.autograd.grad(                    # 求梯度
                 loss.requires_grad_(True),
                 [latents],
                 retain_graph=True,
             )[0]
             latents = (
                 latents - grad_cond * self.scheduler.sigmas[i] ** 2
             )
```

那么这个主程序部分也就结束了。但是，loss 函数中 Attention Map 这部分是怎么提取出来的呢？

## Attention Map

```python
        self.attention_store = AttentionStore() # 一个存储 Attention map 的地方
        self.register_attention_control()
```

主程序中代码先用这两句确定了一个 attention_store，然后用一个函数“注册”。这个函数在 AAE 的 pipeline 里，接下来就可以在 Unet 运作的时候直接读出 attention map，这是怎么做到的？让我们来看看。

> 这里就要理解 python 的特性了：def 中 list 作为形参的话，形参的改变会导致实参的改变（和 C 一样）。以及，a=b 这样的 list 赋值其实是对象引用的赋值，a 变了 b 也会跟着变。知道这两个特性以后我们就可以理解这样的“实时更新”是怎么做到的了。

register_attention_control 是 AAE pipeline 的一个函数，被 layout 的 pipeline 继承了。

```python
    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count
```

这里面调用了 unet.attn_processors，我们先看这个函数：

#### function: attn_processors

```python
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors
```

> @property 装饰器是使得函数可以和属性一样被访问，比如外层调用的时候就没有加括号。

named_children 是 torch.nn.Module 的函数，可以返回一个字典，包含模型的所有子模块和其名字，这是 GPT 给的例子：

```python
import torch
import torch.nn as nn
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = SimpleModel()
for name, child in model.named_children():
    print(name, child)
'''
输出为：
fc1 Linear(in_features=10, out_features=5, bias=True)
fc2 Linear(in_features=5, out_features=1, bias=True)
'''
```

值得注意的是，模块是基于 nn.Module 的实例，你可以自定义模块，那么 named_children 也会返回你自定义模块的类。那么这段代码也可以理解了。因为 Unet 包含了很多子模块，子模块里也有很多子模块（比如一个下采样模块包含了 resnet、transformer2d 等），所以这个函数用递归的形式遍历所有子模块。如果出现了 set_processor 的 attribution，那么说明这是 attention 模块，那么就把它加入到一个字典里，表示需要读取。现在我们看哪些子模块有 set_processor，就可以知道 attention 存在哪里。

~~但是我 tm 没找到。所以我打开了远程算力把这东西 print 出来。~~

`'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor': <diffusers.models.attention_processor.AttnProcessor object at 0x7f0b03228790>`

是存在一个 down_blocks 的数组里，然后里面有 attentions 的数组，然后里面是 transformer_blocks 的数组，然后里面有内置了 attn，

```python
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers)
            ]
        )
```

所以这个函数的功能就是，把 unet 中所有有 attention 的部分扒出来，放在一个字典里给你。

#### Attention and its processer

我们扒出 Attention 在哪里。在一个 BasicTransformerBlock 的 class 里，里面调用了 Attention。这个 Attention 来自库里实现的 attention_processor。

```python
        self.attn1 = Attention(
            query_dim=dim,                             # Q 的通道数
            heads=num_attention_heads,                 # multihead-attention 里的 head 数 
            dim_head=attention_head_dim,               # 一个 head 的通道数，也就是 K 和 Q 的通道数
            dropout=dropout,                           # dropout 正则化的概率，默认 0
            bias=attention_bias,                       # 全连接网络加不加 bias
            cross_attention_dim=cross_attention_dim if only_cross_attention else None, # encoder(CLIP) 编码后每一个 token 的通道数
            upcast_attention=upcast_attention,
        )
```

> 每次前向传播时，以概率 dropout 将输入张量的一些元素设置为零，其余元素按比例缩放。这样可以在模型训练过程中随机地“丢弃”一些神经元的输出，以减少神经网络的过拟合程度，从而提高模型的泛化能力。

然后我点进去大致浏览了一下，能看到很多之前了解过的东西（不是）。其用一个叫做 processor 的对象来完成 forward 操作。他的 forward 函数就是直接调用...

```python
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
```

其中，to_q 函数是 nn.Module.Linear(query_dim, dim_head * heads)

to_k, to_v 函数是 nn.Module.Linear(cross_attention_dim, dim_head * heads)

最基本的 processor 是 AttnProcessor，代码是这样的：

```python
class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,              # 图像编码后的空间，也就是要成为 Q 的空间
        encoder_hidden_states=None, # prompt 编码后的空间，也就是要成为 K V 的空间
        attention_mask=None,
        temb=None,
    ):
        # 多头注意力机制
```

现在重点看这个函数:

```python
    def set_processor(self, processor: "AttnProcessor"):
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor
```

相当于提供了更换 processor 的一个接口。（真妙啊！）

#### function: set_attn_processor

```python
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)
```

那么就是把 Unet 的所有 attention 的 processor 全换成给定的。

#### function: register_attention_control

我们回到一开始的 register 函数：

```python
    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys(): # 把存在 attention 的拔下来
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet) # 换成给定的 processor

        self.unet.set_attn_processor(attn_procs)                    # 递归修改
        self.attention_store.num_att_layers = cross_att_count
``` 

#### class: AttendExciteAttnProcessor

可以发现，这就是一层包装，然后把 AttendExciteAttnProcessor 创建出来，全部替换。至于 AttendExciteAttnProcessor 的实现，很简单，甚至删掉了 diffusers 源码的特判。

这里也着重介绍一下 **head_to_batch_dim** 和 **batch_to_head_dim**。

```python
# 关于这两个函数，是 multihead-attention 的，单独放在这里方便理解
 
    def head_to_batch_dim(self, tensor, out_dim=3): # 经过了 to_* 以后，末端 dim 包含了 head 个头的部分
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size) # 所以我们把 head 单独拎出来
        tensor = tensor.permute(0, 2, 1, 3) # 然后把 head 扔到前面
        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size) # 把 head 融合进 batch_size 里
        return tensor

    def batch_to_head_dim(self, tensor): 
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim) # 之前 head 融进 batch_size 了，现在分离出来
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size) # head 融进最后一层
        return tensor

class AttendExciteAttnProcessor:
    def __init__(self, attnstore, place_in_unet): # attenstore 和 place_in_unet
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states # 不是交叉的话，自己和自己做
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if attention_probs.requires_grad: # 存储 attention map
            self.attnstore(attention_probs, is_cross, self.place_in_unet)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) # 过全连接
        hidden_states = attn.to_out[1](hidden_states) # dropout 正则化
        return hidden_states
```

attention map 存在了 self.attnstore，也就是 pipeline 类的 self.attention_store 里，我们继续看 attention_store。这是外面定义的类了！终于到最外层了！

#### AttentionStore

```python
class AttentionStore:
    @staticmethod # 不用传 self 的修饰器
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str): # 从 processor 中获得 unet 的位置、是否是 cross_attention 的信息
        if is_cross:
            if attn.shape[1] in self.attn_res:
                self.step_store[place_in_unet].append(attn) # 存进字典

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers: # 如果满了，那么存入答案，当前清空（感觉没必要这么麻烦）
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def maps(self, block_type: str):
        return self.attention_store[block_type] # 读取 attention map

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=[256, 64]):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store() # step_store 存当前 step 正在做的，attention_store 存已完成的 step 的
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res
```

这是一个缓存的包装。我们来看看 loss 部分吧！

#### Loss

```python
    def _compute_loss(self, token_indices, bboxes, device) -> torch.Tensor:
        loss = 0
        object_number = len(bboxes)
        total_maps = 0
        for location in ["mid", "up"]: # 论文里说的，只用考虑 mid 层和 up 层。这个接口其实是测试用的。
            for attn_map_integrated in self.attention_store.maps(location):
                attn_map = attn_map_integrated.chunk(2)[1]

                b, i, j = attn_map.shape
                H = W = int(np.sqrt(i))

                total_maps += 1
                for obj_idx in range(object_number): # 枚举 box
                    obj_loss = 0
                    obj_box = bboxes[obj_idx]

                    x_min, y_min, x_max, y_max = ( # 本来正则化的，变成像素
                        obj_box[0] * W,
                        obj_box[1] * H,
                        obj_box[2] * W,
                        obj_box[3] * H,
                    )
                    mask = torch.zeros((H, W), device=device)
                    mask[round(y_min) : round(y_max), round(x_min) : round(x_max)] = 1

                    for obj_position in token_indices[obj_idx]: # 枚举 box 内的物体
                        ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W) # attention map 第一维是 batch_size * head，第二维是图像 Q，第三维是 token KV
                        activation_value = (ca_map_obj * mask).reshape(b, -1).sum(
                            dim=-1
                        ) / ca_map_obj.reshape(b, -1).sum(dim=-1) # 算出 masked 的 / 所有的

                        obj_loss += torch.mean((1 - activation_value) ** 2)

                    loss += obj_loss / len(token_indices[obj_idx])

        loss /= object_number * total_maps
        return loss
```

至此，我们把整个逻辑框架都看明白了。