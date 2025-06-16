# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py
import copy
import inspect
import math
import re
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    vae_decode,
    vae_encode,
)
from ltx_video.models.transformers.symmetric_patchifier import Patchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import TimestepShifter
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.utils.prompt_enhance_utils import generate_cinematic_prompt
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.autoencoders.vae_encode import (
    un_normalize_latents,
    normalize_latents,
)


try:
    import torch_xla.distributed.spmd as xs
except ImportError:
    xs = None

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

#定义宽度与高度比例左边比例对应右边宽度和高度，指定宽高比后从列表只自动选择最接近的标准分辨率
ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}


# 检索时间步骤
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    skip_initial_inference_steps: int = 0,
    skip_final_inference_steps: int = 0,
    **kwargs,
):
    """
    调用调度程序的"set_timesteps"方法 并在调用后从调度程序中检索时间步长，处理自定定义时间步长任何kwargs都将提供给"scheduler.set_timesteps".

    参数:
        scheduler (`SchedulerMixin`):
            要从中获取时间步长的程序.
        num_inference_steps (`int`):
            使用预训练模型生成样本时的扩散步骤,如果使用
            `timesteps` 必须为 `None`.
        device (`str` or `torch.device`, *可选*):
            时间步长应用到移动设备如果为None则不会移动时间步长.
        timesteps (`List[int]`, *optional*):
            用于支持时间步长的任意间距的自定定义时间步长,如果为None，则默认的使用调度器的时间步长间隔策略如果传递了timesteps,
            则传递num_inference_steps必须为None`.
        max_timestep ('float', *optional*, defaults to 1.0):
            图像到图像/视频到视频的初始噪声级别，如果timestamp将为truncated以大于或等于this的时间戳开始

    返回:
        `元组[torch.Tensor, int]`: 一个元组,其中第一个元素来自调度器的时间步长调度,需第二个元素是推理步骤的数量.
    """
    # 如果时间步长是没有
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        logger.debug(f"✅管道文件时间步长调度")

        if (
            skip_initial_inference_steps < 0
            or skip_final_inference_steps < 0
            or skip_initial_inference_steps + skip_final_inference_steps
            >= num_inference_steps
        ):
            raise ValueError(
                "invalid skip inference step values: must be non-negative and the sum of skip_initial_inference_steps and skip_final_inference_steps must be less than the number of inference steps"
            )

        timesteps = timesteps[
            skip_initial_inference_steps : len(timesteps) - skip_final_inference_steps
        ]
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        num_inference_steps = len(timesteps)
        logger.debug(f"✅管道文件时间步长步骤")

    return timesteps, num_inference_steps


@dataclass
# 类调节项
class ConditioningItem:
    """
    定义单个帧调节项 - 单个帧或帧序列。
    属性:
        media_item (torch.Tensor): shape=(b, 3, f, h, w). 张量：要设置条件的媒体项.
        media_frame_number (int): 生成视频媒体项的起始帧号.
        conditioning_strength (float): 调节的强度 (1.0 = 完全调节).
        media_x (Optional[int]): 生成帧中媒体项的可选左x坐标.
        media_y (Optional[int]): 生成帧中媒体项的可选项部y坐标.
    """

    # torch.Tensor 是一种包含单一数据类型元素的多维矩阵
    media_item: torch.Tensor
    media_frame_number: int
    conditioning_strength: float
    media_x: Optional[int] = None
    media_y: Optional[int] = None

# 视频生成管道
class LTXVideoPipeline(DiffusionPipeline):
    r"""
    使用LTX-Video生成文本到图像的管道.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    参数:
        vae ([`AutoencoderKL`]):
            变分自编码器 (VAE) 模型， 用于将图像编码与解码为潜在表示或从潜在表示中解码图像.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. This uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`Transformer2DModel`]):
            一个文本条件`Transformer2DModel` 对编码的图像潜在值进行降噪.
        scheduler ([`SchedulerMixin`]):
            与`transformer` 结合使用调度程序，用于对编码的图像潜在值进行降噪.
    """
    # 再编译
    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    # 可选组件
    _optional_components = [
        "tokenizer",
        "text_encoder",
        "prompt_enhancer_image_caption_model",
        "prompt_enhancer_image_caption_processor",
        "prompt_enhancer_llm_model",
        "prompt_enhancer_llm_tokenizer",
    ]
    # 模型卸载顺序
    model_cpu_offload_seq = "prompt_enhancer_image_caption_model->prompt_enhancer_llm_model->text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,                                  # t5模型分词器
        text_encoder: T5EncoderModel,                            # t5模型文本编码
        vae: AutoencoderKL,                                      # 主模型权重自带
        transformer: Transformer3DModel,                         # 主模型权重文件
        scheduler: DPMSolverMultistepScheduler,                  # 多步骤调度器，用于采样
        patchifier: Patchifier,
        prompt_enhancer_image_caption_model: AutoModelForCausalLM,   # 图像增强编码
        prompt_enhancer_image_caption_processor: AutoProcessor,      # 图像增强标题处理器
        prompt_enhancer_llm_model: AutoModelForCausalLM,             # 提示词增强模型
        prompt_enhancer_llm_tokenizer: AutoTokenizer,                # 提示词增强分词器
        allowed_inference_steps: Optional[List[float]] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            patchifier=patchifier,
            prompt_enhancer_image_caption_model=prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor=prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model=prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer=prompt_enhancer_llm_tokenizer,
        )

        self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
            self.vae
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.allowed_inference_steps = allowed_inference_steps

    def mask_text_embeddings(self, emb, mask):
        if emb.shape[0] == 1:
            keep_index = mask.sum().item()
            return emb[:, :, :keep_index, :], keep_index
        else:
            masked_feature = emb * mask[:, None, :, None]
            return masked_feature, emb.shape[2]

    # 文本编码提示词
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],                                              # 提示词
        do_classifier_free_guidance: bool = True,                                   # 是否使用隐式分类器(classifier_free)引导
        negative_prompt: str = "",                                                  # 负面提示词
        num_images_per_prompt: int = 1,                                             # 每一个提示生成图像数
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,                          # 提示词嵌入
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,                 # 负面提示词嵌入
        prompt_attention_mask: Optional[torch.FloatTensor] = None,                  # 提示注意掩码
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,         # 负面提示注意掩码
        text_encoder_max_tokens: int = 256,                                         # 文本编码最大令牌
        **kwargs,
    ):
        r"""
        将提示编码为文本编码器隐藏状态.

        参数:
            prompt (`str` or `List[str]`, *optional*):
                提示词进行编码
            negative_prompt (`str` or `List[str]`, *optional*):
                提示不指导图像生成，如果不定义则必须传递"negative_prompt_embeds",相反，不使用"guidance"时忽略（即,如果"guidance_scale"小于"1"则忽略)
                为这应该是""。
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                是否使用classifier free 进行引导
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                每个提示生成的图像数
            device: (`torch.device`, *optional*):
                torch device 放置生成的嵌入
            prompt_embeds (`torch.FloatTensor`, *optional*):
                预先生成的文本嵌入. 可用于轻松调整文本输入, *例如.* 提示符权重. 如果不是
                前提是, 将从`prompt` 输入参数生成文本嵌入.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                预先生成的反正提示文本嵌入.
        """
        # 警告信息mask_feature已被弃用，也不影响计算，在1.0.0版本后删除
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        if device is None:                                                  # 如果设备没有指定
            device = self._execution_device                                 # 将使用self._execution_device指定的设备
            logger.debug(f"✅管道执行设备: {device}")



        if prompt is not None and isinstance(prompt, str):                  # 如果提示词没有和is实例（提示词, 字符串）
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):                # 如果没有提示词和实例（提示词, 列表）
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            logger.debug(f"✅提示词: {batch_size}") 

        # 文本序列的最大长度
        max_length = (
            text_encoder_max_tokens  # TPU 仅支持 128的倍数长度
        )
        if prompt_embeds is None:                                               # 如果没有提示词嵌入
            assert (
                self.text_encoder is not None
            ), "You should provide either prompt_embeds or self.text_encoder should not be None,"
            text_enc_device = next(self.text_encoder.parameters()).device
            prompt = self._text_preprocessing(prompt)                            # 提示词文本预处理
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",                                            # 填充为最大序列长度
                max_length=max_length,
                truncation=True,                                                 # 截断为真
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, max_length - 1 : -1]
                )

                logger.debug(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )
                logger.debug(f"✅以下部份内容初截断因为CLIP模型最多只能处理: {max_length}个词: {removed_text}")

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(text_enc_device)
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(text_enc_device), attention_mask=prompt_attention_mask
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        # 将提示嵌入转移到设备
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # 获取原始嵌入张量的形状信息：原始大小,序列长度,嵌入维度 =
        bs_embed, seq_len, _ = prompt_embeds.shape
        # 复制文本嵌入：为每个提示生成多个图像,使用repeat方法沿第二个维度复制张量(0维不变、1维序列维度、2维嵌入维度不变)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # 重塑张量维度以匹配批量处理要求
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        # 同样复制注意力掩码
        prompt_attention_mask = prompt_attention_mask.repeat(1, num_images_per_prompt)
        # 重塑注意力掩码维度
        prompt_attention_mask = prompt_attention_mask.view(
            bs_embed * num_images_per_prompt, -1
        )

        # 获取无条件嵌入以进行无分类器指导
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = self._text_preprocessing(negative_prompt)
            uncond_tokens = uncond_tokens * batch_size
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                text_enc_device
            )

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(text_enc_device),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # 使用MPS友好方法为每个提示符的每一代重复无条件嵌入
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(
                1, num_images_per_prompt
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                bs_embed * num_images_per_prompt, -1
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        # ====== 新增文本投影层开始 ======
        # 确保投影层存在（动态创建）
        if not hasattr(self, 'text_projection'):
            # 从transformer配置获取caption_channels
            caption_channels = self.transformer.config.caption_channels

        # 创建投影层 (768->4096)
        self.text_projection = nn.Linear(
            768, 
            caption_channels,
            device=device,
            dtype=prompt_embeds.dtype if prompt_embeds is not None else torch.float32
        )

        # 将投影层注册为模块以便设备管理
        self.text_projection = self.text_projection.to(device)
        logger.debug(f"✅创建文本投影层: 768 -> {caption_channels}")

        # 应用投影到正提示
        if prompt_embeds is not None:
            logger.debug(f"投影前正提示维度: {prompt_embeds.shape}")
            prompt_embeds = self.text_projection(prompt_embeds)
            logger.debug(f"投影后正提示维度: {prompt_embeds.shape}")

        # 应用投影到负提示
        if negative_prompt_embeds is not None:
            logger.debug(f"投影前负提示维度: {negative_prompt_embeds.shape}")
            negative_prompt_embeds = self.text_projection(negative_prompt_embeds)
            logger.debug(f"投影后负提示维度: {negative_prompt_embeds.shape}")
        # ====== 新增文本投影层结束 ======

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    # 准备额外的步骤kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # 为scheduler 步骤准备额外的kwargs, 因为所有的 schedulers 都具有相同的签名
        # eta (η)仅用于DDIMScheduler, 其它调度器忽略它.
        # eta 对应的论文 η in DDIM paper: https://arxiv.org/abs/2010.02502
        # 并且应在 [0, 1]之间

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受 generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        # 从这部份开始修改，确保所有张量都在同一设备上
        if accepts_generator:                              
            if generator is not None:                              # 增加代码：确保生成器在正确设备上
                generator = generator.to(self._execution_device)   # 增加关键修复代码：将生成器转移到当前设备
            extra_step_kwargs["generator"] = generator

        # 确保所有值都在正确设备上
        for key in extra_step_kwargs:                               # 新增代码
            if isinstance(extra_step_kwargs[key], torch.Tensor):    # 新增代码
                extra_step_kwargs[key] = extra_step_kwargs[key].to(self._execution_device)   # 新增代码

        return extra_step_kwargs

    # 检查输入
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        enhance_prompt=False,
    ):

        if height % 8 != 0 or width % 8 != 0:
            logger.error(f"❌ height` and `width` 必须被 8 整除，但他们是 {height} and {width}") 
            raise ValueError(
                f"`height` and `width` 必须被 8 整除，但他们是 {height} and {width}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"无法转发两个 `prompt`: {提示} and `prompt_embeds`: {prompt_embeds}. 请确保"
                " 只转发其中一个."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            logger.error(f"❌ prompt` has to be of type `str` or `list` but is {type(prompt)}") 
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if prompt is not None and negative_prompt_embeds is not None:
            logger.error(f"❌ 失败: {prompt}, {negative_prompt_embeds}") 
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            logger.error(f"❌ 失败: {negative_prompt}, {negative_prompt_embeds}") 
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError(
                "Must provide `prompt_attention_mask` when specifying `prompt_embeds`."
            )

        if (
            negative_prompt_embeds is not None
            and negative_prompt_attention_mask is None
        ):
            raise ValueError(
                "Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                logger.error(f"❌ 失败: {prompt_embeds.shape},{negative_prompt_embeds.shape}") 
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                logger.error(f"❌ 失败: {prompt_attention_mask.shape}, {negative_prompt_attention_mask.shape} ") 
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

        # 如果图像增强提示
        if enhance_prompt:
            assert (
                self.prompt_enhancer_image_caption_model is not None
            ), "Image caption model must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_image_caption_processor is not None
            ), "Image caption processor must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_llm_model is not None
            ), "Text prompt enhancer model must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_llm_tokenizer is not None
            ), "Text prompt enhancer tokenizer must be initialized if enhance_prompt is True"


    # 文本预处理
    def _text_preprocessing(self, text):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            text = text.strip()
            return text

        return [process(t) for t in text]

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        将依赖于时间步长的噪声添加到硬调节潜在噪声中.
        这有助于运动的连续性，尤其是单个帧为条件时.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        latents: torch.Tensor | None,
        media_items: torch.Tensor | None,
        timestep: float,
        latent_shape: torch.Size | Tuple[Any, ...],
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | List[torch.Generator],
        vae_per_channel_normalize: bool = True,
    ):
        """
        准备要去噪的初始潜在张量.
        The latents are either pure noise or a noised version of the encoded media items.
        参数:
            latents (`torch.FloatTensor` or `None`):
                The latents to use (provided by the user) or `None` to create new latents.
            media_items (`torch.FloatTensor` or `None`):
                An image or video to be updated using img2img or vid2vid. The media item is encoded and noised.
            timestep (`float`):
                The timestep to noise the encoded media_items to.
            latent_shape (`torch.Size`):
                目标潜在的形状.
            dtype (`torch.dtype`):
                The target dtype.
            device (`torch.device`):
                目标设备.
            generator (`torch.Generator` or `List[torch.Generator]`):
                Generator(s) to be used for the noising process.
            vae_per_channel_normalize ('bool'):
                When encoding the media_items, whether to normalize the latents per-channel.
        Returns:
            `torch.FloatTensor`: The latents to be used for the denoising process. This is a tensor of shape
            (batch_size, num_channels, height, width).
        """
        if isinstance(generator, list) and len(generator) != latent_shape[0]:
            logger.debug(
                f"你传递了一个长度(length) {len(generator)}, 但请求的是有效批处理"
                f" 大小 {latent_shape[0]}. 确保批次大小与生成器匹配."
            )

        # 使用给定的latents 或编码器的媒体项 初始化latent
        assert (
            latents is None or media_items is None
        ), "Cannot provide both latents and media_items. Please provide only one of the two."

        assert (
            latents is None and media_items is None or timestep < 1.0
        ), "Input media_item or latents are provided, but they will be replaced with noise."

        if media_items is not None:
            media_items = media_items.to(dtype=self.vae.dtype, device=self.vae.device)
            logger.debug(f"✅prepare_latents: media_items.device = {media_items.device}, vae.device = {self.vae.device}")
            assert media_items.device == self.vae.device, f"Media_items not on vae.device: {media_items.device} vs {self.vae.device}"

            latents = vae_encode(
                media_items.to(dtype=self.vae.dtype, device=self.vae.device),
                self.vae,
                vae_per_channel_normalize=vae_per_channel_normalize,
            )
        if latents is not None:
            assert (
                latents.shape == latent_shape
            ), f"Latents have to be of shape {latent_shape} but are {latents.shape}."
            latents = latents.to(device=device, dtype=dtype)

        # For backward compatibility, generate in the "patchified" shape and rearrange
        b, c, f, h, w = latent_shape
        noise = randn_tensor(
            (b, f * h * w, c), generator=generator, device=device, dtype=dtype
        )
        noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)
        logger.debug(f"✅噪声张量设备: {noise.device}")

        # scale the initial noise by the standard deviation required by the scheduler
        noise = noise * self.scheduler.init_noise_sigma

        if latents is None:
            latents = noise
        else:
            # Noise the latents to the required (first) timestep
            latents = timestep * noise + (1 - timestep) * latents

        return latents

    @staticmethod
    def classify_height_width_bin(
        height: int, width: int, ratios: dict
    ) -> Tuple[int, int]:
        """返回分箱高度与宽度."""
        ar = float(height / width)
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
        default_hw = ratios[closest_ratio]

        return int(default_hw[0]), int(default_hw[1])

    @staticmethod
    def resize_and_crop_tensor(
        samples: torch.Tensor, new_width: int, new_height: int
    ) -> torch.Tensor:
        n_frames, orig_height, orig_width = samples.shape[-3:]

        # 检查是否需要调整大小
        if orig_height != new_height or orig_width != new_width:
            ratio = max(new_height / orig_height, new_width / orig_width)
            resized_width = int(orig_width * ratio)
            resized_height = int(orig_height * ratio)

            # 调整大小
            samples = LTXVideoPipeline.resize_tensor(
                samples, resized_height, resized_width
            )

            # 中心裁剪
            start_x = (resized_width - new_width) // 2
            end_x = start_x + new_width
            start_y = (resized_height - new_height) // 2
            end_y = start_y + new_height
            samples = samples[..., start_y:end_y, start_x:end_x]

        return samples

    @staticmethod
    # 调整张量
    def resize_tensor(media_items, height, width):
        n_frames = media_items.shape[2]
        if media_items.shape[-2:] != (height, width):
            media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
            media_items = F.interpolate(
                media_items,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            media_items = rearrange(media_items, "(b n) c h w -> b c n h w", n=n_frames)

        return media_items

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        timesteps: List[int] = None,
        guidance_scale: Union[float, List[float]] = 4.5,
        cfg_star_rescale: bool = False,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        skip_block_list: Optional[Union[List[List[int]], List[int]]] = None,
        stg_scale: Union[float, List[float]] = 1.0,
        rescaling_scale: Union[float, List[float]] = 0.7,
        guidance_timesteps: Optional[List[int]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        conditioning_items: Optional[List[ConditioningItem]] = None,
        decode_timestep: Union[List[float], float] = 0.0,
        decode_noise_scale: Optional[List[float]] = None,
        mixed_precision: bool = False,
        offload_to_cpu: bool = False,
        enhance_prompt: bool = False,
        text_encoder_max_tokens: int = 256,
        stochastic_sampling: bool = False,
        media_items: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        调用pipeline 进行生成时的函数.

        参数:
            prompt (`str` or `List[str]`, *optional*):
                用于指导影像生成的提示，如果未定义，则必须传递原始提示词(prompt_embeds).
                相反.
            negative_prompt (`str` or `List[str]`, *optional*):
                反向提示词The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                T降噪步骤数. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. If `timesteps` is provided, this parameter is ignored.
            skip_initial_inference_steps (`int`, *optional*, defaults to 0):
                要跳过的时间步骤. After calculating the timesteps, this number of timesteps will
                be removed from the beginning of the timesteps list. Meaning the highest-timesteps values will not run.
            skip_final_inference_steps (`int`, *optional*, defaults to 0):
                要跳过的最终时间步数. After calculating the timesteps, this number of timesteps will
                be removed from the end of the timesteps list. Meaning the lowest-timesteps values will not run.
            timesteps (`List[int]`, *optional*):
                用于降噪过程中的自定义时间步长. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            cfg_star_rescale (`bool`, *optional*, defaults to `False`):
                如果设置为 `True`, 则应用 CFG 星形重缩放. Scales the negative prediction according to dot
                product between positive and negative.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                每个提示生成的图像数.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                生成图像高度.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                生成图像宽度.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                预先生成的文本嵌入. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.FloatTensor`, *optional*): 为文本嵌入预先生成的注意力掩码.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                负面提示词文本嵌入. This negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                为负面提示生成注意力掩码.
            output_type (`str`, *optional*, defaults to `"pil"`):
                生成图像的格式. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            enhance_prompt (`bool`, *optional*, defaults to `False`):
                If set to `True`, the prompt is enhanced using a LLM model.
            text_encoder_max_tokens (`int`, *optional*, defaults to `256`):
                用于文本编码器的最在令牌数，默认为256.
            stochastic_sampling (`bool`, *optional*, defaults to `False`):
                如果设置为 `True`, 则采样是随机的. 如果设置为 `False`, 则采样是确定性的.
            media_items ('torch.Tensor', *optional*):
                用于图像到图像/视频到视频的输入媒体项.
        例子:

        返回:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        is_video = kwargs.get("is_video", False)
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )
        logger.debug(f"✅is_video: 高度{height}, 宽度{width}")

        # 2. transformer的默认高度与宽度
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        self.video_scale_factor = self.video_scale_factor if is_video else 1
        vae_per_channel_normalize = kwargs.get("vae_per_channel_normalize", True)
        image_cond_noise_scale = kwargs.get("image_cond_noise_scale", 0.0)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_num_frames = num_frames // self.video_scale_factor
        if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
            latent_num_frames += 1
        latent_shape = (
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        )
        logger.debug(f"✅默认高度与宽度: 高度{latent_height}, 宽度{latent_width}")

        # 准备降噪时间步长列表

        retrieve_timesteps_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_timesteps_kwargs["samples_shape"] = latent_shape

        assert (
            skip_initial_inference_steps == 0
            or latents is not None
            or media_items is not None
        ), (
            f"skip_initial_inference_steps ({skip_initial_inference_steps}) is used for image-to-image/video-to-video - "
            "media_item or latents should be provided."
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            skip_initial_inference_steps=skip_initial_inference_steps,
            skip_final_inference_steps=skip_final_inference_steps,
            **retrieve_timesteps_kwargs,
        )

        if self.allowed_inference_steps is not None:
            for timestep in [round(x, 4) for x in timesteps.tolist()]:
                assert (
                    timestep in self.allowed_inference_steps
                ), f"Invalid inference timestep {timestep}. Allowed timesteps are {self.allowed_inference_steps}."

        if guidance_timesteps:
            guidance_mapping = []
            for timestep in timesteps:
                indices = [
                    i for i, val in enumerate(guidance_timesteps) if val <= timestep
                ]
                # assert len(indices) > 0, f"No guidance timestep found for {timestep}"
                guidance_mapping.append(
                    indices[0] if len(indices) > 0 else (len(guidance_timesteps) - 1)
                )
                logger.debug(f"✅管道文件时间步长: {indices}")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        if not isinstance(guidance_scale, List):
            guidance_scale = [guidance_scale] * len(timesteps)
        else:
            guidance_scale = [
                guidance_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        # For simplicity, we are using a constant num_conds for all timesteps, so we need to zero
        # out cases where the guidance scale should not be applied.
        guidance_scale = [x if x > 1.0 else 0.0 for x in guidance_scale]

        if not isinstance(stg_scale, List):
            stg_scale = [stg_scale] * len(timesteps)
        else:
            stg_scale = [stg_scale[guidance_mapping[i]] for i in range(len(timesteps))]

        if not isinstance(rescaling_scale, List):
            rescaling_scale = [rescaling_scale] * len(timesteps)
        else:
            rescaling_scale = [
                rescaling_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        do_classifier_free_guidance = any(x > 1.0 for x in guidance_scale)
        do_spatio_temporal_guidance = any(x > 0.0 for x in stg_scale)
        do_rescaling = any(x != 1.0 for x in rescaling_scale)

        num_conds = 1
        if do_classifier_free_guidance:
            num_conds += 1
        if do_spatio_temporal_guidance:
            num_conds += 1

        # 如果需要将单个列表转换为列表
        if skip_block_list is not None:
            # Convert single list to list of lists if needed
            if len(skip_block_list) == 0 or not isinstance(skip_block_list[0], list):
                skip_block_list = [skip_block_list] * len(timesteps)
            else:
                new_skip_block_list = []
                for i, timestep in enumerate(timesteps):
                    new_skip_block_list.append(skip_block_list[guidance_mapping[i]])
                skip_block_list = new_skip_block_list

        # 准备跳过图层蒙版
        skip_layer_masks: Optional[List[torch.Tensor]] = None
        if do_spatio_temporal_guidance:
            if skip_block_list is not None:
                skip_layer_masks = [
                    self.transformer.create_skip_layer_mask(
                        batch_size, num_conds, num_conds - 1, skip_blocks
                    )
                    for skip_blocks in skip_block_list
                ]

        if enhance_prompt:
            self.prompt_enhancer_image_caption_model = (
                self.prompt_enhancer_image_caption_model.to(self._execution_device)
            )
            self.prompt_enhancer_llm_model = self.prompt_enhancer_llm_model.to(
                self._execution_device
            )

            prompt = generate_cinematic_prompt(
                self.prompt_enhancer_image_caption_model,
                self.prompt_enhancer_image_caption_processor,
                self.prompt_enhancer_llm_model,
                self.prompt_enhancer_llm_tokenizer,
                prompt,
                conditioning_items,
                max_new_tokens=text_encoder_max_tokens,
            )

        # 3. 对输入的提示进行编码
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            text_encoder_max_tokens=text_encoder_max_tokens,
        )

        if offload_to_cpu and self.text_encoder is not None:
            self.text_encoder = self.text_encoder.cpu()

        self.transformer = self.transformer.to(self._execution_device)

        prompt_embeds_batch = prompt_embeds
        prompt_attention_mask_batch = prompt_attention_mask
        if do_classifier_free_guidance:
            prompt_embeds_batch = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            prompt_attention_mask_batch = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )
        if do_spatio_temporal_guidance:
            prompt_embeds_batch = torch.cat([prompt_embeds_batch, prompt_embeds], dim=0)
            prompt_attention_mask_batch = torch.cat(
                [
                    prompt_attention_mask_batch,
                    prompt_attention_mask,
                ],
                dim=0,
            )

        # 4. 使用提供的介质和调节项准备初始 latent

        # 准备初台潜在张量, shape = (b, c, f, h, w)
        latents = self.prepare_latents(
            latents=latents,
            media_items=media_items,
            timestep=timesteps[0],
            latent_shape=latent_shape,
            dtype=prompt_embeds_batch.dtype,
            device=device,
            generator=generator,
            vae_per_channel_normalize=vae_per_channel_normalize,
        )

        # 用条件项更新latents 并将他们修补成(b, n, c)
        latents, pixel_coords, conditioning_mask, num_cond_latents = (
            self.prepare_conditioning(
                conditioning_items=conditioning_items,
                init_latents=latents,
                num_frames=num_frames,
                height=height,
                width=width,
                vae_per_channel_normalize=vae_per_channel_normalize,
                generator=generator,
            )
        )
        init_latents = latents.clone()  # 用于 image_cond_noise_update

        pixel_coords = torch.cat([pixel_coords] * num_conds)
        orig_conditioning_mask = conditioning_mask
        if conditioning_mask is not None and is_video:
            assert num_images_per_prompt == 1
            conditioning_mask = torch.cat([conditioning_mask] * num_conds)
        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

        # 6. 准备额外的步骤
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. 去噪循环
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if conditioning_mask is not None and image_cond_noise_scale > 0.0:
                    latents = self.add_noise_to_image_conditioning_latents(
                        t,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        orig_conditioning_mask,
                        generator,
                    )

                latent_model_input = (
                    torch.cat([latents] * num_conds) if num_conds > 1 else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: 这需要 CPU 和 GPU同步. 如果可以，请尝试将时间步长作为张量传递
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).unsqueeze(-1)

                if conditioning_mask is not None:
                    # Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
                    # and will start to be denoised when the current timestep is lower than their conditioning timestep.
                    current_timestep = torch.min(
                        current_timestep, 1.0 - conditioning_mask
                    )

                # 根据 `mixed_precision`选择合适的上下文管理器
                if mixed_precision:
                    context_manager = torch.autocast(device.type, dtype=torch.bfloat16)
                else:
                    context_manager = nullcontext()  # Dummy context manager

                # 预测噪声 model_output
                with context_manager:
                    logger.debug(f"✅模型输入设备: {latent_model_input.device}")
                    logger.debug(f"✅坐标输入设备: {fractional_coords.device}")
                    logger.debug(f"✅提示词嵌入设备: {prompt_embeds_batch.device}")
                    assert latent_model_input.device == self._execution_device
                    assert fractional_coords.device == self._execution_device
                    assert prompt_embeds_batch.device == self._execution_device

                    noise_pred = self.transformer(
                        latent_model_input.to(self.transformer.dtype),
                        indices_grid=fractional_coords,
                        encoder_hidden_states=prompt_embeds_batch.to(
                            self.transformer.dtype
                        ),
                        encoder_attention_mask=prompt_attention_mask_batch,
                        timestep=current_timestep,
                        skip_layer_mask=(
                            skip_layer_masks[i]
                            if skip_layer_masks is not None
                            else None
                        ),
                        skip_layer_strategy=skip_layer_strategy,
                        return_dict=False,
                    )[0]

                # 执行指导
                if do_spatio_temporal_guidance:
                    noise_pred_text, noise_pred_text_perturb = noise_pred.chunk(
                        num_conds
                    )[-2:]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_conds)[:2]

                    if cfg_star_rescale:
                        # Rescales the unconditional noise prediction using the projection of the conditional prediction onto it:
                        # α = (⟨ε_text, ε_uncond⟩ / ||ε_uncond||²), then ε_uncond ← α * ε_uncond
                        # where ε_text is the conditional noise prediction and ε_uncond is the unconditional one.
                        positive_flat = noise_pred_text.view(batch_size, -1)
                        negative_flat = noise_pred_uncond.view(batch_size, -1)
                        dot_product = torch.sum(
                            positive_flat * negative_flat, dim=1, keepdim=True
                        )
                        squared_norm = (
                            torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
                        )
                        alpha = dot_product / squared_norm
                        noise_pred_uncond = alpha * noise_pred_uncond

                    noise_pred = noise_pred_uncond + guidance_scale[i] * (
                        noise_pred_text - noise_pred_uncond
                    )
                elif do_spatio_temporal_guidance:
                    noise_pred = noise_pred_text
                if do_spatio_temporal_guidance:
                    noise_pred = noise_pred + stg_scale[i] * (
                        noise_pred_text - noise_pred_text_perturb
                    )
                    if do_rescaling and stg_scale[i] > 0.0:
                        noise_pred_text_std = noise_pred_text.view(batch_size, -1).std(
                            dim=1, keepdim=True
                        )
                        noise_pred_std = noise_pred.view(batch_size, -1).std(
                            dim=1, keepdim=True
                        )

                        factor = noise_pred_text_std / noise_pred_std
                        factor = rescaling_scale[i] * factor + (1 - rescaling_scale[i])

                        noise_pred = noise_pred * factor.view(batch_size, 1, 1)

                current_timestep = current_timestep[:1]
                # 学习 sigma
                if (
                    self.transformer.config.out_channels // 2
                    == self.transformer.config.in_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # 计算上一个镜像: x_t -> x_t-1
                latents = self.denoising_step(
                    latents,
                    noise_pred,
                    current_timestep,
                    orig_conditioning_mask,
                    t,
                    extra_step_kwargs,
                    stochastic_sampling=stochastic_sampling,
                )

                # 调用回调, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, t, {})

        if offload_to_cpu:
            self.transformer = self.transformer.cpu()
            if self._execution_device == "cuda":
                torch.cuda.empty_cache()

        # 添加删除的 conditioning latents
        latents = latents[:, num_cond_latents:]

        latents = self.patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            out_channels=self.transformer.in_channels
            // math.prod(self.patchifier.patch_size),
        )
        if output_type != "latent":
            if self.vae.decoder.timestep_conditioning:
                noise = torch.randn_like(latents)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * latents.shape[0]
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * latents.shape[0]

                decode_timestep = torch.tensor(decode_timestep).to(latents.device)
                decode_noise_scale = torch.tensor(decode_noise_scale).to(
                    latents.device
                )[:, None, None, None, None]
                latents = (
                    latents * (1 - decode_noise_scale) + noise * decode_noise_scale
                )
            else:
                decode_timestep = None
            image = vae_decode(
                latents,
                self.vae,
                is_video,
                vae_per_channel_normalize=kwargs["vae_per_channel_normalize"],
                timestep=decode_timestep,
            )

            image = self.image_processor.postprocess(image, output_type=output_type)
            logger.debug(f"✅视频生成管道处理后的图像: {image }") 

        else:
            image = latents

        # 卸载所有模型
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def denoising_step(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        current_timestep: torch.Tensor,
        conditioning_mask: torch.Tensor,
        t: float,
        extra_step_kwargs,
        t_eps=1e-6,
        stochastic_sampling=False,
    ):
        """
        Perform the denoising step for the required tokens, based on the current timestep and
        conditioning mask:
        Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
        and will start to be denoised when the current timestep is equal or lower than their
        conditioning timestep.
        (hard-conditioning latents with conditioning_mask = 1.0 are never denoised)
        """
        logger.debug(f"✅去噪输入潜在变量设备: {latents.device}")
        logger.debug(f"✅噪声预测设备: {noise_pred.device}")

        # Denoise the latents using the scheduler
        denoised_latents = self.scheduler.step(
            noise_pred,
            t if current_timestep is None else current_timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
            stochastic_sampling=stochastic_sampling,
        )[0]

        if conditioning_mask is None:
            return denoised_latents

        tokens_to_denoise_mask = (t - t_eps < (1.0 - conditioning_mask)).unsqueeze(-1)
        return torch.where(tokens_to_denoise_mask, denoised_latents, latents)

    # 准备条件
    def prepare_conditioning(
        self,
        conditioning_items: Optional[List[ConditioningItem]],
        init_latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        vae_per_channel_normalize: bool = False,
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        根据提供的条件贰准备 conditioning tokens.

        此方法将提供条件项 (视频帧或单帧) 编码为 latents
        并将他们与初始潜在张量（latent tensor）集成. 他还会计算相应的像素
        坐标, 一个表示潜在的掩码，经及调节潜在物.

        参数:
            conditioning_items (Optional[List[ConditioningItem]]): ConditioningItem 对象列表.
            init_latents (torch.Tensor): 形在为(b, c, f_l, h_l, w_l)的初始潜在张量, 其中
                `f_l` 是潜在张量的数量, `h_l` 和 `w_l` 是潜在空间维度.
            num_frames, height, width: 生成视频的尺寸.
            vae_per_channel_normalize (bool, optional): 是否在VAE编码期间对通道进行归一化.
                默认`False`.
            generator: 随机生成器

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - `init_latents` (torch.Tensor): The updated latent tensor including conditioning latents,
                  patchified into (b, n, c) shape.
                - `init_pixel_coords` (torch.Tensor): The pixel coordinates corresponding to the updated
                  latent tensor.
                - `conditioning_mask` (torch.Tensor): A mask indicating the conditioning-strength of each
                  latent token.
                - `num_cond_latents` (int): The total number of latent tokens added from conditioning items.

        Raises:
            AssertionError: If input shapes, dimensions, or conditions for applying conditioning are invalid.
        """
        assert isinstance(self.vae, CausalVideoAutoencoder)

        if conditioning_items:
            batch_size, _, num_latent_frames = init_latents.shape[:3]

            init_conditioning_mask = torch.zeros(
                init_latents[:, 0, :, :, :].shape,
                dtype=torch.float32,
                device=init_latents.device,
            )
            logger.debug(f"✅条件掩码设备: {init_conditioning_mask.device}")

            extra_conditioning_latents = []
            extra_conditioning_pixel_coords = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0  # Number of extra conditioning latents added (should be removed before decoding)

            # 处理每个条件项
            for conditioning_item in conditioning_items:
                conditioning_item = self._resize_conditioning_item(
                    conditioning_item, height, width
                )
                media_item = conditioning_item.media_item
                media_frame_number = conditioning_item.media_frame_number
                strength = conditioning_item.conditioning_strength
                assert media_item.ndim == 5  # (b, c, f, h, w)
                b, c, n_frames, h, w = media_item.shape
                assert (
                    height == h and width == w
                ) or media_frame_number == 0, f"Dimensions do not match: {height}x{width} != {h}x{w} - allowed only when media_frame_number == 0"
                logger.debug(f"✅管道文件获取高度宽度条件: 高度={height}x{width}, 媒体尺寸={h}x{w}")
                assert n_frames % 8 == 1
                assert (
                    media_frame_number >= 0
                    and media_frame_number + n_frames <= num_frames
                )

                # 对提供的conditioning 媒体项进行编码
                media_item_latents = vae_encode(
                    media_item.to(dtype=self.vae.dtype, device=self.vae.device),
                    self.vae,
                    vae_per_channel_normalize=vae_per_channel_normalize,
                ).to(dtype=init_latents.dtype)
                logger.debug(f"✅条件项更新: media_item_latents.device = {media_item_latents.device}, vae.device = {self.vae.device}")
                assert media_item_latents.device == self.vae.device, f"media_item_latents.device mismatch: {media_item_latents.device} vs {self.vae.device}"

                # 处理不同条件情况
                if media_frame_number == 0:
                    # 获取潜在条件项的目标空间位置
                    media_item_latents, l_x, l_y = self._get_latent_spatial_position(
                        media_item_latents,
                        conditioning_item,
                        height,
                        width,
                        strip_latent_border=True,
                    )
                    b, c_l, f_l, h_l, w_l = media_item_latents.shape


                    # 第一帧或第一帧序列 - 只需更新初始噪声潜伏物与掩码
                    init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l] = (
                        torch.lerp(
                            init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l],
                            media_item_latents,
                            strength,
                        )
                    )
                    init_conditioning_mask[
                        :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l
                    ] = strength
                else:
                    # Non-first frame or sequence
                    if n_frames > 1:
                        # Handle non-first sequence.
                        # Encoded latents are either fully consumed, or the prefix is handled separately below.
                        (
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                        ) = self._handle_non_first_conditioning_sequence(
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                            media_frame_number,
                            strength,
                        )


                    # 单帧或序列前缀潜在值
                    if media_item_latents is not None:
                        noise = randn_tensor(
                            media_item_latents.shape,
                            generator=generator,
                            device=media_item_latents.device,
                            dtype=media_item_latents.dtype,
                        )

                        media_item_latents = torch.lerp(
                            noise, media_item_latents, strength
                        )

                        # 修补额外的条件潜在变量并计算他们的像素坐标
                        media_item_latents, latent_coords = self.patchifier.patchify(
                            latents=media_item_latents
                        )
                        pixel_coords = latent_to_pixel_coords(
                            latent_coords,
                            self.vae,
                            causal_fix=self.transformer.config.causal_temporal_positioning,
                        )
                        logger.debug(f"✅像素坐标: {pixel_coords}")

                        # 更新帧号以匹配目标帧号
                        pixel_coords[:, 0] += media_frame_number
                        extra_conditioning_num_latents += media_item_latents.shape[1]

                        conditioning_mask = torch.full(
                            media_item_latents.shape[:2],
                            strength,
                            dtype=torch.float32,
                            device=init_latents.device,
                        )

                        extra_conditioning_latents.append(media_item_latents)
                        extra_conditioning_pixel_coords.append(pixel_coords)
                        extra_conditioning_mask.append(conditioning_mask)


        # 修补更新潜在值并计算他们的像素坐标
        init_latents, init_latent_coords = self.patchifier.patchify(
            latents=init_latents
        )
        init_pixel_coords = latent_to_pixel_coords(
            init_latent_coords,
            self.vae,
            causal_fix=self.transformer.config.causal_temporal_positioning,
        )

        if not conditioning_items:
            return init_latents, init_pixel_coords, None, 0

        init_conditioning_mask, _ = self.patchifier.patchify(
            latents=init_conditioning_mask.unsqueeze(1)
        )
        init_conditioning_mask = init_conditioning_mask.squeeze(-1)

        if extra_conditioning_latents:
            # 堆叠额外的条件latents像素坐标与掩码，这此列表中张量其中一个仍是 CPU，其它是 cuda，就会抛出遇到的错误。
            init_latents = torch.cat([*extra_conditioning_latents, init_latents], dim=1)
            init_pixel_coords = torch.cat(
                [*extra_conditioning_pixel_coords, init_pixel_coords], dim=2
            )
            init_conditioning_mask = torch.cat(
                [*extra_conditioning_mask, init_conditioning_mask], dim=1
            )

            if self.transformer.use_tpu_flash_attention:
                # When flash attention is used, keep the original number of tokens by removing
                #   tokens from the end.
                init_latents = init_latents[:, :-extra_conditioning_num_latents]
                init_pixel_coords = init_pixel_coords[
                    :, :, :-extra_conditioning_num_latents
                ]
                init_conditioning_mask = init_conditioning_mask[
                    :, :-extra_conditioning_num_latents
                ]

        return (
            init_latents,
            init_pixel_coords,
            init_conditioning_mask,
            extra_conditioning_num_latents,
        )


    @staticmethod
    def _resize_conditioning_item(
        conditioning_item: ConditioningItem,
        height: int,
        width: int,
    ):
        if conditioning_item.media_x or conditioning_item.media_y:
            raise ValueError(
                "Provide media_item in the target size for spatial conditioning."
            )
        new_conditioning_item = copy.copy(conditioning_item)
        new_conditioning_item.media_item = LTXVideoPipeline.resize_tensor(
            conditioning_item.media_item, height, width
        )
        return new_conditioning_item


    # 获取条件项在潜在空间的空间位置
    def _get_latent_spatial_position(
        self,
        latents: torch.Tensor,
        conditioning_item: ConditioningItem,
        height: int,
        width: int,
        strip_latent_border,
    ):
        """
        Get the spatial position of the conditioning item in the latent space.
        If requested, strip the conditioning latent borders that do not align with target borders.
        (border latents look different then other latents and might confuse the model)
        """
        scale = self.vae_scale_factor
        h, w = conditioning_item.media_item.shape[-2:]
        assert (
            h <= height and w <= width
        ), f"Conditioning item size {h}x{w} is larger than target size {height}x{width}"
        assert h % scale == 0 and w % scale == 0

        # 计算媒体项的开始与结束空间位置
        x_start, y_start = conditioning_item.media_x, conditioning_item.media_y
        x_start = (width - w) // 2 if x_start is None else x_start
        y_start = (height - h) // 2 if y_start is None else y_start
        x_end, y_end = x_start + w, y_start + h
        assert (
            x_end <= width and y_end <= height
        ), f"Conditioning item {x_start}:{x_end}x{y_start}:{y_end} is out of bounds for target size {width}x{height}"

        if strip_latent_border:
            # Strip one latent from left/right and/or top/bottom, update x, y accordingly
            if x_start > 0:
                x_start += scale
                latents = latents[:, :, :, :, 1:]

            if y_start > 0:
                y_start += scale
                latents = latents[:, :, :, 1:, :]

            if x_end < width:
                latents = latents[:, :, :, :, :-1]

            if y_end < height:
                latents = latents[:, :, :, :-1, :]

        return latents, x_start // scale, y_start // scale


    @staticmethod
    def _handle_non_first_conditioning_sequence(
        init_latents: torch.Tensor,
        init_conditioning_mask: torch.Tensor,
        latents: torch.Tensor,
        media_frame_number: int,
        strength: float,
        num_prefix_latent_frames: int = 2,
        prefix_latents_mode: str = "concat",
        prefix_soft_conditioning_strength: float = 0.15,
    ):
        """
        Special handling for a conditioning sequence that does not start on the first frame.
        The special handling is required to allow a short encoded video to be used as middle
        (or last) sequence in a longer video.
        Args:
            init_latents (torch.Tensor): The initial noise latents to be updated.
            init_conditioning_mask (torch.Tensor): The initial conditioning mask to be updated.
            latents (torch.Tensor): The encoded conditioning item.
            media_frame_number (int): The target frame number of the first frame in the conditioning sequence.
            strength (float): The conditioning strength for the conditioning latents.
            num_prefix_latent_frames (int, optional): The length of the sequence prefix, to be handled
                separately. Defaults to 2.
            prefix_latents_mode (str, optional): Special treatment for prefix (boundary) latents.
                - "drop": Drop the prefix latents.
                - "soft": Use the prefix latents, but with soft-conditioning
                - "concat": Add the prefix latents as extra tokens (like single frames)
            prefix_soft_conditioning_strength (float, optional): The strength of the soft-conditioning for
                the prefix latents, relevant if `prefix_latents_mode` is "soft". Defaults to 0.1.

        """
        f_l = latents.shape[2]
        f_l_p = num_prefix_latent_frames
        assert f_l >= f_l_p
        assert media_frame_number % 8 == 0
        if f_l > f_l_p:
            # Insert the conditioning latents **excluding the prefix** into the sequence
            f_l_start = media_frame_number // 8 + f_l_p
            f_l_end = f_l_start + f_l - f_l_p
            init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                init_latents[:, :, f_l_start:f_l_end],
                latents[:, :, f_l_p:],
                strength,
            )
            # Mark these latent frames as conditioning latents
            init_conditioning_mask[:, f_l_start:f_l_end] = strength
            logger.debug(f"✅管道文件完整数据: {strength}")

        # Handle the prefix-latents
        if prefix_latents_mode == "soft":
            if f_l_p > 1:
                # Drop the first (single-frame) latent and soft-condition the remaining prefix
                f_l_start = media_frame_number // 8 + 1
                f_l_end = f_l_start + f_l_p - 1
                strength = min(prefix_soft_conditioning_strength, strength)
                init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                    init_latents[:, :, f_l_start:f_l_end],
                    latents[:, :, 1:f_l_p],
                    strength,
                )
                # Mark these latent frames as conditioning latents
                init_conditioning_mask[:, f_l_start:f_l_end] = strength
            latents = None  # No more latents to handle
        elif prefix_latents_mode == "drop":
            # Drop the prefix latents
            latents = None
        elif prefix_latents_mode == "concat":
            # Pass-on the prefix latents to be handled as extra conditioning frames
            latents = latents[:, :, :f_l_p]
        else:
            raise ValueError(f"Invalid prefix_latents_mode: {prefix_latents_mode}")
        return (
            init_latents,
            init_conditioning_mask,
            latents,
        )


    def trim_conditioning_sequence(
        self, start_frame: int, sequence_num_frames: int, target_num_frames: int
    ):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.

        Returns:
            int: updated sequence length
        """
        scale_factor = self.video_scale_factor
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        logger.debug(f"✅管道文件完整数据: {num_frames}")
        return num_frames


def adain_filter_latent(
    latents: torch.Tensor, reference_latents: torch.Tensor, factor=1.0
):
    """
    Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on
    statistics from a reference latent tensor.

    Args:
        latent (torch.Tensor): Input latents to normalize
        reference_latent (torch.Tensor): The reference latents providing style statistics.
        factor (float): Blending factor between original and transformed latent.
                       Range: -10.0 to 10.0, Default: 1.0

    Returns:
        torch.Tensor: The transformed latent tensor
    """
    result = latents.clone()

    for i in range(latents.size(0)):
        for c in range(latents.size(1)):
            r_sd, r_mean = torch.std_mean(
                reference_latents[i, c], dim=None
            )  # index by original dim order
            i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

            result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

    result = torch.lerp(latents, result, factor)
    logger.debug(f"✅管道文件完整数据: {result}")
    return result

# 视频生成步骤流程
class LTXMultiScalePipeline:
    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        assert latents.device == latest_upsampler.device

        latents = un_normalize_latents(
            latents, self.vae, vae_per_channel_normalize=True
        )
        upsampled_latents = latest_upsampler(latents)
        upsampled_latents = normalize_latents(
            upsampled_latents, self.vae, vae_per_channel_normalize=True
        )
        return upsampled_latents

    def __init__(
        self, video_pipeline: LTXVideoPipeline, latent_upsampler: LatentUpsampler
    ):
        self.video_pipeline = video_pipeline
        self.vae = video_pipeline.vae
        self.latent_upsampler = latent_upsampler


    def __call__(
        self,
        downscale_factor: float,
        first_pass: dict,
        second_pass: dict,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_kwargs = kwargs.copy()
        original_output_type = kwargs["output_type"]
        original_width = kwargs["width"]
        original_height = kwargs["height"]

        x_width = int(kwargs["width"] * downscale_factor)
        downscaled_width = x_width - (x_width % self.video_pipeline.vae_scale_factor)
        x_height = int(kwargs["height"] * downscale_factor)
        downscaled_height = x_height - (x_height % self.video_pipeline.vae_scale_factor)
        logger.debug(f"✅管道文件获取目标高宽值: 宽{x_width}, 高{x_height}")

        kwargs["output_type"] = "latent"
        kwargs["width"] = downscaled_width
        kwargs["height"] = downscaled_height
        kwargs.update(**first_pass)
        result = self.video_pipeline(*args, **kwargs)
        latents = result.images

        upsampled_latents = self._upsample_latents(self.latent_upsampler, latents)
        upsampled_latents = adain_filter_latent(
            latents=upsampled_latents, reference_latents=latents
        )

        kwargs = original_kwargs

        kwargs["latents"] = upsampled_latents
        kwargs["output_type"] = original_output_type
        kwargs["width"] = downscaled_width * 2
        kwargs["height"] = downscaled_height * 2
        kwargs.update(**second_pass)

        result = self.video_pipeline(*args, **kwargs)
        if original_output_type != "latent":
            num_frames = result.images.shape[2]
            videos = rearrange(result.images, "b c f h w -> (b f) c h w")

            videos = F.interpolate(
                videos,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            videos = rearrange(videos, "(b f) c h w -> b c f h w", f=num_frames)
            logger.debug(f"✅管道文件完整数据: {videos}")
            result.images = videos

        return result
