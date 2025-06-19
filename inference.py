import argparse
import os
import random
import logging
from datetime import datetime
from pathlib import Path

from typing import Optional, List, Union
import yaml

import imageio
import json
import numpy as np
import torch
import cv2
from safetensors import safe_open
from transformers import BitsAndBytesConfig
from PIL import Image
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from huggingface_hub import hf_hub_download

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    ConditioningItem,
    LTXVideoPipeline,
    LTXMultiScalePipeline,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
import ltx_video.pipelines.crf_compressor as crf_compressor

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

logger = logging.getLogger("LTX-Video")

def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)    # 计算设备总显存有多少
        logger.debug(f"✅设备总显存: {total_memory}") 
        return total_memory
    return 0

def get_device():
    if torch.cuda.is_available():                           # 如果"cuda"可以获得
        return "cuda"                         # 返回英伟达设备
    elif torch.backends.mps.is_available():                 # 如果torch后端计划可发获得，返回计划，则当前 PyTorch 版本支持使用 NVIDIA MPS 来加速训练。
        return "mps"                          # 返回苹果设备
    return "cpu"

def load_image_to_tensor_with_resize_and_crop(
    image_input: Union[str, Image.Image],
    target_height: int = 512,
    target_width: int = 768,
    just_crop: bool = False,
) -> torch.Tensor:
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be either a file path or a PIL Image object")
        logger.debug(f"✅图像文件存在: {image}")

    input_width, input_height = image.size                     # 获取原始图像的宽度和高度
    aspect_ratio_target = target_width / target_height         # 计算生成图像的宽高比（目标宽度除以目标高度）
    aspect_ratio_frame = input_width / input_height            # 计算原始图像尺寸的宽高比（原始宽度除以原始高度）
    logger.debug(f"✅原始图像宽高比: {aspect_ratio_frame:.4f} (≈{input_width}:{input_height})")
    logger.debug(f"✅目标宽高比: {aspect_ratio_target:.4f} (≈{target_width}:{target_height})")
    if aspect_ratio_frame > aspect_ratio_target:               # 原始图像比目标更宽（横图）
        new_width = int(input_height * aspect_ratio_target)    # 裁剪后的尺寸
        new_height = input_height
        x_start = (input_width - new_width) // 2               # 裁剪区域的起始坐标：//2水平居中
        y_start = 0
    else:                                                       # 原始图像比目标更高（竖图）
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)     # 裁剪后的尺寸
        x_start = 0
        y_start = (input_height - new_height) // 2              # 垂直居中

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    if not just_crop:
        image = image.resize((target_width, target_height))
        logger.debug(f"✅最终图像尺寸: {image}")

    image = np.array(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    frame_tensor = torch.from_numpy(image).float()
    frame_tensor = frame_tensor.to('cuda:0')
    logger.debug(f"✅frame_tensor(from_numpy) device: {frame_tensor.device}")

    frame_tensor = crf_compressor.compress(frame_tensor / 255.0) * 255.0
    logger.debug(f"✅frame_tensor(after compress) device: {frame_tensor.device}")

    frame_tensor = frame_tensor.permute(2, 0, 1)
    logger.debug(f"✅frame_tensor(after permute) device: {frame_tensor.device}")

    frame_tensor = (frame_tensor / 127.5) - 1.0
    logger.debug(f"✅frame_tensor(after normalize) device: {frame_tensor.device}")

    result_tensor = frame_tensor.unsqueeze(0).unsqueeze(2)
    logger.debug(f"✅frame_tensor(final 5D) device: {result_tensor.device}")
    return result_tensor

def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    pad_height = target_height - source_height
    pad_width = target_width - source_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    logger.debug(f"✅图像填充的完整数据: {padding}")
    return padding

def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # 删除非字母并转换为小写
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    words = clean_text.split()

    result = []
    current_length = 0

    for word in words:
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)

# 生成输出视频名称
def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )

def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def main():
    # 创建解析器定义需要的参数自动生成帮助和使用信息使用 add_argument() 方法向解析器中添加参数
    parser = argparse.ArgumentParser(
        description="Load models from separate directories and run the pipeline."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="保存输出视频的文件夹路径, 如果 None 将保存在 outputs/ 目录中.",
    )
    parser.add_argument("--seed", type=int, default="171198")

    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="每个提示生成的图像数量",
    )
    parser.add_argument(
        "--image_cond_noise_scale",
        type=float,
        default=0.15,
        help="添加到条件图像上的噪声量",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="输出视频帧的高度（如果提供了输入图像，则为可选参数）.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1216,
        help="输出视频帧的宽度 （如果没有将从图像中推断）.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=121,
        help="要生成的视频帧数",
    )
    parser.add_argument(
        "--frame_rate", type=int, default=30, help="输出视频的帧率"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="运行推理的设备. 如果没有指定, 将自动检测使用 CUDA 或 MPS （如果可用）, 否则使用 CPU.",
    )
    parser.add_argument(
        "--pipeline_config",
        type=str,
        default="configs/ltxv-2b-0.9.5.yaml",
        help="pipeline配置文件的路径, 其中包含pipeline的参数",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="用于指导生成的文本提示词",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="对不需要的功能进行否定提示（负面提示词）",
    )
    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="将不必要的计算卸载到 CPU.",
    )
    parser.add_argument(
        "--input_media_path",
        type=str,
        default=None,
        help="要使用视频到视频pipeline修改的输入视频（或图像）的路径",
    )
    parser.add_argument(
        "--conditioning_media_paths",
        type=str,
        nargs="*",
        help="调节媒体 (图像或视频)的路径列表. 每条路径将用作调节项.",
    )
    parser.add_argument(
        "--conditioning_strengths",
        type=float,
        nargs="*",
        help="每个调节项调节强度列表 (介于 0 和 1 之间) 必须与调节项的数量匹配.",
    )
    parser.add_argument(
        "--conditioning_start_frames",
        type=int,
        nargs="*",
        help="应用每个调节项的帧索引列表.必须与调节项的数量匹配.",
    )

    args = parser.parse_args()
    logger.warning(f"Running generation with arguments: {args}")
    infer(**vars(args))

def get_supported_precision():
    # 判断设备是否支持 bfloat16，否则用 float16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32

# 定义视频生成的管道
def create_ltx_video_pipeline(
    ckpt_path: str,                         # 权重路径
    precision: str,                         # 精度
    text_encoder_model_name_or_path: str,   # 文本编码器路径
    sampler: Optional[str] = None,          # 采样、自选（没有）
    device: Optional[str] = None,           # 设备、自选（没有）
    enhance_prompt: bool = False,
    prompt_enhancer_image_caption_model_name_or_path: Optional[str] = None,          # 提示图像增强模型路径
    prompt_enhancer_llm_model_name_or_path: Optional[str] = None,                    # 提示词增强模型路径
) -> LTXVideoPipeline:
    ckpt_path = Path(ckpt_path)
    assert os.path.exists(
        ckpt_path
    ), f"Ckpt path provided (--ckpt_path) {ckpt_path} does not exist"

    with safe_open(ckpt_path, framework="pt") as f:
        metadata = f.metadata()
        config_str = metadata.get("config")
        configs = json.loads(config_str)
        allowed_inference_steps = configs.get("allowed_inference_steps", None)

    # ====== 修改: 分片加载与低显存适配 ======
    # 自动选择 dtype
    dtype = get_supported_precision()
    # 自动判断是否8G卡，采用分片/CPU+GPU混合
    total_gpu_mem = get_total_gpu_memory()
    is_low_mem_gpu = (total_gpu_mem > 0 and total_gpu_mem <= 8)

    # 使用device_map和low_cpu_mem_usage参数进行分片加载
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if is_low_mem_gpu:
        # 8G卡采用分片加载
        load_kwargs["device_map"] = "auto"
    else:
        # 大卡直接全加载到GPU
        load_kwargs["device_map"] = {"": device or get_device()}

    # ===== 关键: from_pretrained 传递分片参数 =====
    try:
        vae = CausalVideoAutoencoder.from_pretrained(str(ckpt_path), **load_kwargs)                 # 加载自动编码器
        logger.debug(f"✅VAE模型加载成功: {ckpt_path}, device: {vae.device if hasattr(vae, 'device') else 'unknown'}")
    except Exception as e:
        logger.error(f"❌VAE模型加载失败：{ckpt_path}, error: {e}")
    try:
        transformer = Transformer3DModel.from_pretrained(str(ckpt_path), **load_kwargs)             # 加载注意力机制模型transformer（含编码器与解码器）
        logger.debug(f"✅Transformer模型加载成功: {ckpt_path}, device: {transformer.device if hasattr(transformer, 'device') else 'unknown'}")
    except Exception as e:
        logger.error(f"❌Transformer模型加载失败：{ckpt_path}, error: {e}")

    # 如果指定了采样器则使用checkpoint，否则从本地加载采样器
    try:
        if sampler == "from_checkpoint" or not sampler:                                            
            scheduler = RectifiedFlowScheduler.from_pretrained(str(ckpt_path))
            logger.debug(f"✅采样器Scheduler加载成功: {ckpt_path}")
        else:
            scheduler = RectifiedFlowScheduler(
                sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
            )
            logger.debug(f"✅采样器Scheduler加载成功(自定义): {ckpt_path}")
    except Exception as e:
        logger.error(f"❌采样器Scheduler加载失败：{ckpt_path}, error: {e}")

    # 文本编碼器从T5模型加载
    try:
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_model_name_or_path, subfolder="text_encoder")
        logger.debug(
            f"✅文本编码模型加载成功: {text_encoder_model_name_or_path}, "
            f"device: {text_encoder.device if hasattr(text_encoder, 'device') else 'unknown'}"
        )
    except Exception as e:
        logger.error(f"❌文本编码模型加载失败: {text_encoder_model_name_or_path}, error: {e}")

    patchifier = SymmetricPatchifier(patch_size=1)      # 修补器

    try: 
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_model_name_or_path, subfolder="tokenizer")
        logger.debug(f"✅分词器加载成功: {text_encoder_model_name_or_path}")
    except Exception as e:
        logger.error(f"❌分词器加载失败: {text_encoder_model_name_or_path}, error: {e}")

    transformer = transformer.to(device)      # 将注意力机制模型transformer转移到设备上
    logger.debug(f"✅transformer.to({device}) device: {next(transformer.parameters()).device}")

    vae = vae.to(device)                      # 将自动编码器转移到设备上
    logger.debug(f"✅vae.to({device}) device: {next(vae.parameters()).device}")

    text_encoder = text_encoder.to(device)    # 将文本编码器转移到设备上
    logger.debug(f"✅text_encoder.to({device}) device: {next(text_encoder.parameters()).device}")

    if enhance_prompt:
        try:
            prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained( 
                prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
            )
            logger.debug(
                f"✅图像增强模型加载成功: {prompt_enhancer_image_caption_model_name_or_path},"
                f"device: {prompt_enhancer_image_caption_model.device if hasattr(prompt_enhancer_image_caption_model, 'device') else 'unknown'}"
            )
        except Exception as e:
            logger.error(f"❌图像增强模型加载失败: {prompt_enhancer_image_caption_model_name_or_path}, error: {e}")

        try:
            prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(  
                prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
            )
            logger.debug(f"✅图像增强处理器加载成功: {prompt_enhancer_image_caption_model_name_or_path}")
        except Exception as e:
            logger.error(f"❌图像增强处理器加载失败: {prompt_enhancer_image_caption_model_name_or_path}, error: {e}")

        try:
            prompt_enhancer_llm_model = AutoModelForCausalLM.from_pretrained( 
                prompt_enhancer_llm_model_name_or_path,
                torch_dtype="bfloat16",
            )
            logger.debug(
                f"✅提示词增强LLM语言模型加载成功: {prompt_enhancer_llm_model_name_or_path}," 
                f"device: {prompt_enhancer_llm_model.device if hasattr(prompt_enhancer_llm_model, 'device') else 'unknown'}"
            )
        except Exception as e:
            logger.error(f"❌提示词增强LLM语言模型加载失败: {prompt_enhancer_llm_model_name_or_path}, error: {e}")

        try:
            prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained( 
                prompt_enhancer_llm_model_name_or_path,
            )
            logger.debug(f"✅提示词增强LLM分词器加载成功: {prompt_enhancer_llm_model_name_or_path}")
        except Exception as e:
            logger.error(f"❌提示词增强LLM分词器加载失败: {prompt_enhancer_llm_model_name_or_path}, error: {e}")

    else:
        prompt_enhancer_image_caption_model = None
        prompt_enhancer_image_caption_processor = None
        prompt_enhancer_llm_model = None
        prompt_enhancer_llm_tokenizer = None

    # ==== 修改开始: 自动适配精度 ====
    dtype = get_supported_precision()  # 新增
    vae = vae.to(dtype)                # 修改
    if precision in ["bfloat16", "float16"]:
        transformer = transformer.to(dtype)   # 修改
    text_encoder = text_encoder.to(dtype)     # 修改

    # 将子模型用于pipeline，创建子模型字典
    submodel_dict = {
        "transformer": transformer,                               # 将注意力机制模型
        "patchifier": patchifier,                                 # 修补器
        "text_encoder": text_encoder,                             # 文本编码器
        "tokenizer": tokenizer,                                   # 分词器
        "scheduler": scheduler,                                   # 调度程序
        "vae": vae,                                               # 变分自编码器
        "prompt_enhancer_image_caption_model": prompt_enhancer_image_caption_model,                # 图像增强模型
        "prompt_enhancer_image_caption_processor": prompt_enhancer_image_caption_processor,        # 图像增强模型处理器
        "prompt_enhancer_llm_model": prompt_enhancer_llm_model,                                    # 提示词增强模型
        "prompt_enhancer_llm_tokenizer": prompt_enhancer_llm_tokenizer,                            # 提示词增强模型分词器
        "allowed_inference_steps": allowed_inference_steps,                                        # 允许的推理步数
    }

    pipeline = LTXVideoPipeline(**submodel_dict)      # 管道等于视频生成子模型字典
    pipeline = pipeline.to(device)                    # 将管道转移到设备上
    logger.debug(f"✅管道pipeline动行设备: {device}")
    return pipeline                                    # 返回管道

def create_latent_upsampler(latent_upsampler_model_path: str, device: str):
    # 动态上采样器：采样器模型路径
    try:
        latent_upsampler = LatentUpsampler.from_pretrained(latent_upsampler_model_path)
        logger.debug(f"✅上采样器加载成功: {latent_upsampler}") 
        latent_upsampler.to(device) 
        logger.debug(f"✅上采样器加载到设备: {device}")
    except Exception as e:
        logger.error(f"❌上采样器加载失败: {latent_upsampler_model_path}, error: {e}")

    latent_upsampler.eval()
    return latent_upsampler

# 定义推断
def infer(
    output_path: Optional[str],                                        # 输出路径：自选
    seed: int,                                                         # 种子
    pipeline_config: str,                                              # 管道配置文件路径
    image_cond_noise_scale: float,                                     # 图像条件
    height: Optional[int],                                             # 图像高：自选
    width: Optional[int],
    num_frames: int,
    frame_rate: int,
    prompt: str,                                                       # 提示词
    negative_prompt: str,                                              # 反向提示词
    offload_to_cpu: bool,                                              # 将计算任务转移到cpu：Offload 技术将GPU显存中的权重卸载到CPU内存
    input_media_path: Optional[str] = None,                            # 输入媒体路径：自选列表
    conditioning_media_paths: Optional[List[str]] = None,              # 条件媒体路径：自选列表             
    conditioning_strengths: Optional[List[float]] = None,              # 条件强度：自选列表
    conditioning_start_frames: Optional[List[int]] = None,             # 条件开始框架：自选输入
    device: Optional[str] = None,                                      # 设备：自选
    **kwargs,                                                          # 关键字参数字典
):
    # 检查 管道配置文件是否为文件
    if not os.path.isfile(pipeline_config):
        raise ValueError(f"Pipeline config file {pipeline_config} does not exist")
    with open(pipeline_config, "r") as f:
        pipeline_config = yaml.safe_load(f)
        logger.debug(f"✅配置文件存在: {pipeline_config}")
                     
    models_dir = "MODEL_DIR"

    ltxv_model_name_or_path = pipeline_config["checkpoint_path"]
    if not os.path.isfile(ltxv_model_name_or_path):
        ltxv_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=ltxv_model_name_or_path,
            local_dir=models_dir,
            repo_type="model",
        )
    else:
        ltxv_model_path = ltxv_model_name_or_path
        logger.debug(f"✅LTX-Video模型存在: {ltxv_model_path}")

    # 从配置文件中加载图像放大模型
    spatial_upscaler_model_name_or_path = pipeline_config.get(
        "spatial_upscaler_model_path"
    )
    if spatial_upscaler_model_name_or_path and not os.path.isfile(
        spatial_upscaler_model_name_or_path
    ):
        spatial_upscaler_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=spatial_upscaler_model_name_or_path,
            local_dir=models_dir,
            repo_type="model",
        )
    else:
        spatial_upscaler_model_path = spatial_upscaler_model_name_or_path
        logger.debug(f"✅推理文件加载图像放大模型存在: {spatial_upscaler_model_path}")

    if kwargs.get("input_image_path", None):             # **kwargs代表关键字参数
        logger.warning(
            "Please use conditioning_media_paths instead of input_image_path."
        )
        assert not conditioning_media_paths and not conditioning_start_frames             # 没有条件媒体路径和没有条件开始框架
        conditioning_media_paths = [kwargs["input_image_path"]]                           # 条件媒体从关键参数中获取图像输入路径
        conditioning_start_frames = [0]
        logger.debug(f"✅图像路径存在: {conditioning_media_paths}")
        logger.debug(f"✅条件框架: {conditioning_start_frames}")

    # 验证条件图像参数
    if conditioning_media_paths:
        # 使用默认强度 1.0
        if not conditioning_strengths:
            conditioning_strengths = [1.0] * len(conditioning_media_paths)
        if not conditioning_start_frames:
            raise ValueError(
                "If `conditioning_media_paths` is provided, "
                "`conditioning_start_frames` must also be provided"
            )
            logger.debug(f"✅图像调节强度: {conditioning_strengths}")

        if len(conditioning_media_paths) != len(conditioning_strengths) or len(
            conditioning_media_paths
        ) != len(conditioning_start_frames):
            raise ValueError(
                "`conditioning_media_paths`, `conditioning_strengths`, "
                "and `conditioning_start_frames` must have the same length"
            )
        if any(s < 0 or s > 1 for s in conditioning_strengths):
            raise ValueError("All conditioning strengths must be between 0 and 1")
        if any(f < 0 or f >= num_frames for f in conditioning_start_frames):
            raise ValueError(
                f"All conditioning start frames must be between 0 and {num_frames-1}"
            )

    # 设置随机数种子
    seed_everething(seed)
    if offload_to_cpu and not torch.cuda.is_available():       # 如果卸载到CPU且没有"cuda"可用
        logger.debug("offload_to_cpu 设置为True, 但不会发生卸载， 因为model 已在 CPU上运行.")
        offload_to_cpu = False                                 # 卸载到cpu为假
    else:
        offload_to_cpu = offload_to_cpu and get_total_gpu_memory() < 30        # 卸载到cpu和获取总数gpu内存小于30
        logger.debug(f"✅gpu内存小于30设备运行在cpu: {offload_to_cpu}")

    output_dir = (
        Path(output_path)
        if output_path
        else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将维度调整为可初 32 整除，帧数（num_frames ）为 (N * 8 + 1)
    height_padded = ((height - 1) // 32 + 1) * 32                            # 高度填充
    width_padded = ((width - 1) // 32 + 1) * 32                              # 度度填充
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1                  # 帖数填充

    padding = calculate_padding(height, width, height_padded, width_padded)
    logger.debug(f"✅推理文件维度填充调整: {height_padded}x{width_padded}x{num_frames_padded}")
    logger.warning(f"警告填充尺寸: {height_padded}x{width_padded}x{num_frames_padded}")

    # 提示词增强阀值
    prompt_enhancement_words_threshold = pipeline_config[
        "prompt_enhancement_words_threshold"
    ]

    prompt_word_count = len(prompt.split())
    enhance_prompt = (
        prompt_enhancement_words_threshold > 0
        and prompt_word_count < prompt_enhancement_words_threshold
    )
    logger.debug(f"✅推理文件增强提示词: {enhance_prompt}")

    if prompt_enhancement_words_threshold > 0 and not enhance_prompt:
        logger.info(
            f"Prompt has {prompt_word_count} words, which exceeds the threshold of {prompt_enhancement_words_threshold}. Prompt enhancement disabled."
        )

    precision = pipeline_config["precision"]                                                       # 精度
    text_encoder_model_name_or_path = pipeline_config["text_encoder_model_name_or_path"]           # 从配置文件中获取文本编码器模型路径
    sampler = pipeline_config["sampler"]                                                           # 从配置文件中获取采样器
    prompt_enhancer_image_caption_model_name_or_path = pipeline_config[
        "prompt_enhancer_image_caption_model_name_or_path"
    ]
    prompt_enhancer_llm_model_name_or_path = pipeline_config[
        "prompt_enhancer_llm_model_name_or_path"
    ]

    # 管道等于视频生成管道构建管道字典
    pipeline = create_ltx_video_pipeline(
        ckpt_path=ltxv_model_path,
        precision=precision,
        text_encoder_model_name_or_path=text_encoder_model_name_or_path,              # 文本编码器模型
        sampler=sampler,
        device=kwargs.get("device", get_device()),  # 从关键参数中获取设备,参数中没有设备指定
        enhance_prompt=enhance_prompt,
        prompt_enhancer_image_caption_model_name_or_path=prompt_enhancer_image_caption_model_name_or_path,
        prompt_enhancer_llm_model_name_or_path=prompt_enhancer_llm_model_name_or_path,
    )
    logger.debug(f"✅推理文件视频生成管道字典设备：{device}")

    if pipeline_config.get("pipeline_type", None) == "multi-scale":
        if not spatial_upscaler_model_path:
            raise ValueError(
                "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
            )
        # 图像放大模型潜在的上采样
        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, pipeline.device
        )
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)
        logger.debug(f"✅推理文件上采样设备：{pipeline.device}")

    logger.debug(f"✅即将推理，全部子模块设备状态：VAE({pipeline.vae.device}), Transformer({pipeline.transformer.device}), TextEncoder({pipeline.text_encoder.device}), ...")

    # 加载图像文件
    media_item = None
    if input_media_path:
        media_item = load_media_file(
            media_path=input_media_path,
            height=height,
            width=width,
            max_frames=num_frames_padded,
            padding=padding,
        )
        logger.debug(f"✅media_item(after load_media_file) device: {media_item.device if media_item is not None else 'None'}")
    
    # infer 调用处
    conditioning_items = (
        prepare_conditioning(
            conditioning_media_paths=conditioning_media_paths,
            conditioning_strengths=conditioning_strengths,
            conditioning_start_frames=conditioning_start_frames,
            height=height,
            width=width,
            num_frames=num_frames,
            padding=padding,
            pipeline=pipeline,
            device=device,  # 新增参数
        )
        if conditioning_media_paths
        else None
    )
    if conditioning_items:   
        for idx, item in enumerate(conditioning_items):  
            logger.debug(f"✅推理: 条件媒体 #{idx} device = {item.media_item.device}, target = {device}")

    # 注意力机制
    stg_mode = pipeline_config.get("stg_mode", "attention_values")
    del pipeline_config["stg_mode"]
    if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")
        logger.debug(f"✅推理文件时空引导模式: {stg_mode}")

    # 为管道准备输入
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
    }

    device = device or get_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    logger.debug(f"✅管道获取设备: {device}, 生成器设备获取{generator}")

    # 图像管道
    images = pipeline(
        **pipeline_config,
        skip_layer_strategy=skip_layer_strategy,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=frame_rate,
        **sample,
        media_items=media_item,
        conditioning_items=conditioning_items,            # 包含CPU张量
        is_video=True,
        vae_per_channel_normalize=True,
        image_cond_noise_scale=image_cond_noise_scale,
        mixed_precision=(precision == "mixed_precision"),
        offload_to_cpu=offload_to_cpu,
        device=device,                                      # cuda:0
        enhance_prompt=enhance_prompt,
    ).images
    logger.debug(f"✅images (from pipeline) device: {images.device if hasattr(images, 'device') else 'Unknown'}")

    # 将填充的图像裁剪为所需的分辨率和帧数
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]
    logger.debug(f"✅images(after crop) device: {images.device}")

    for i in range(images.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        logger.debug(f"✅video_np生成时原images[i] device: {images[i].device}")

        # 将图像非替范化到 [0, 255] 范围
        video_np = (video_np * 255).astype(np.uint8)
        fps = frame_rate
        height, width = video_np.shape[1:3]
        # 如果生成单个图像
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=prompt,
                seed=seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            output_filename = get_unique_filename(
                f"video_output_{i}",
                ".mp4",
                prompt=prompt,
                seed=seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )
            
            # 写入视频
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

        logger.warning(f"Output saved to {output_filename}")
        logger.debug(f"✅输出文件名称: {output_filename}")
    logger.debug(
        f"✅LTXVideoPipeline创建成功，包含子模型: VAE({vae.device}),"
        f"Transformer({transformer.device}), TextEncoder({text_encoder.device}), Scheduler, Tokenizer等"
    )

# 准备条件
def prepare_conditioning(
    conditioning_media_paths: List[str],
    conditioning_strengths: List[float],
    conditioning_start_frames: List[int],
    height: int,
    width: int,
    num_frames: int,
    padding: tuple[int, int, int, int],
    pipeline: LTXVideoPipeline,
    device: str,  # 新增
) -> Optional[List[ConditioningItem]]:
    # 条件参数空列表
    conditioning_items = []
    for path, strength, start_frame in zip(
        conditioning_media_paths, conditioning_strengths, conditioning_start_frames
    ):
        num_input_frames = orig_num_input_frames = get_media_num_frames(path)
        if hasattr(pipeline, "trim_conditioning_sequence") and callable(
            getattr(pipeline, "trim_conditioning_sequence")
        ):
            num_input_frames = pipeline.trim_conditioning_sequence(
                start_frame, orig_num_input_frames, num_frames
            )
        if num_input_frames < orig_num_input_frames:
            logger.warning(
                f"Trimming conditioning video {path} from {orig_num_input_frames} to {num_input_frames} frames."
            )
        
        # 媒体张量=加载的媒体文件
        media_tensor = load_media_file(
            media_path=path,
            height=height,
            width=width,
            max_frames=num_input_frames,
            padding=padding,
            just_crop=True,
        )
        media_tensor = media_tensor.to(device)  # 新增：转到目标设备 
        logger.debug(f"✅media_tensor(after to {device}) device: {media_tensor.device}")
  
        conditioning_items.append(ConditioningItem(media_tensor, start_frame, strength))   # 媒体张量、开始帧、强度
        logger.debug(f"✅媒体张量已转移到设备: {media_tensor.device}") 

    return conditioning_items

# 获取媒体帧参数
def get_media_num_frames(media_path: str) -> int:
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]
    )
    num_frames = 1
    if is_video:
        reader = imageio.get_reader(media_path)
        num_frames = reader.count_frames()
        reader.close()
        logger.debug(f"✅媒体帧参数: {media_num_frames}")
    return num_frames

# 加载媒体文件
def load_media_file(
    media_path: str,
    height: int,
    width: int,
    max_frames: int,
    padding: tuple[int, int, int, int],
    just_crop: bool = False,
) -> torch.Tensor:
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]
    )
    if is_video:
        reader = imageio.get_reader(media_path)
        num_input_frames = min(reader.count_frames(), max_frames)

        # 读取预处理视频中的相关帧.
        frames = []
        for i in range(num_input_frames):
            frame = Image.fromarray(reader.get_data(i))
            frame_tensor = load_image_to_tensor_with_resize_and_crop(
                frame, height, width, just_crop=just_crop
            )
            frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
            frames.append(frame_tensor)
            logger.debug(f"✅媒体帧预处理张量运行在: {frame_tensor.device}") 
        reader.close()

        # 沿时间维度堆叠帧
        media_tensor = torch.cat(frames, dim=2)
    else:                                                                            # 输入图像
        media_tensor = load_image_to_tensor_with_resize_and_crop(
            media_path, height, width, just_crop=just_crop
        )
        logger.debug(f"✅media_tensor(load img) device: {media_tensor.device}")

        media_tensor = torch.nn.functional.pad(media_tensor, padding)
        logger.debug(f"✅media_tensor(after pad) device: {media_tensor.device}")

        media_tensor = media_tensor.to('cuda:0')
        logger.debug(f"✅媒体张量运行在: {media_tensor.device }")
        logger.debug(f"✅media_tensor(final return) device: {media_tensor.device}")

    return media_tensor

if __name__ == "__main__":
    main()
