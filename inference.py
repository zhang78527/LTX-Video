import argparse
import os
import sys
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

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src/gui"))
from src.gui.data_manager import DataManager, MessageTypes

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

logger = logging.getLogger("LTX-Video")

class InferenceModule:
    def __init__(self, config, data_manager, project_root, device):
        self.config = config
        self.data_manager = data_manager
        self.project_root = project_root
        self.device = device
        self._setup_data_manager()
        logger.info("推理模块初始化完成")

    def _setup_data_manager(self):
        """配置数据管理器，监听 INFERENCE_DATA 事件"""
        self.data_manager.register_event(MessageTypes.INFERENCE_DATA, self._inference_data)
        logger.info("推理模块数据管理器配置完成")

    def _inference_data(self, data):
        """接收执行指令任务"""
        if data.get("source") != "generate_video":
            return
        logger.debug(f"✅推理模块接收到数据: {data}")
        inference_data = data.get("data", {})

        params = inference_data.get("params", {})
        config = inference_data.get("config", {})
        conditioning_params = inference_data.get("conditioning_params", {})
        first_pass = inference_data.get("first_pass", {})
        second_pass = inference_data.get("second_pass", {})

        device = params.get("device")
        logger.debug(f"✅推理模块接收到设备信息: {device}")

        # 5. 合并所有参数并推理
        all_params = {
            "device":device,
            **conditioning_params,
            **params, **config,
            "first_pass": first_pass,
            "second_pass": second_pass,
            "pipeline_config": config,
        }

        # 8G显卡强制CPU卸载
        gpu_mem_gb = get_total_gpu_memory()
        if gpu_mem_gb > 0 and gpu_mem_gb <= 8:
            all_params['offload_to_cpu'] = True

        # 主入口直接用 all_params 获取参数和配置
        infer(**all_params)

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
    device: str = "cpu", 
) -> torch.Tensor:
    """加载图像"""
    image = Image.open(image_path).convert("RGB")      # 将图像转换为RGB格式
    input_width, input_height = image.size             # 获取原始图像的宽度和高度
    aspect_ratio_target = target_width / target_height # 计算需生成图像的宽高比（目标宽度除以目标高度）
    aspect_ratio_frame = input_width / input_height    # 计算原始图像尺寸的宽高比（原始宽度除以原始高度）
    logger.debug(f"✅原始图像宽高比: {aspect_ratio_frame:.4f} (≈{input_width}:{input_height})")
    logger.debug(f"✅目标宽高比: {aspect_ratio_target:.4f} (≈{target_width}:{target_height})")
    if aspect_ratio_frame > aspect_ratio_target:       # 原始图像比需生成图像更宽（横图）
        new_width = int(input_height * aspect_ratio_target) # 裁剪后的新宽度
        new_height = input_height                           # 新高度为原高度
        x_start = (input_width - new_width) // 2            # 裁剪区域的起始坐标：//2水平居中
        y_start = 0                                         # 高度起始座标为0
    else:
        new_width = input_width                             # 新宽度与输入宽度一致
        new_height = int(input_width / aspect_ratio_target) # 裁剪后的新高度
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    logger.debug(f"✅裁剪后的图像尺寸: {image}")
    if not just_crop:
        image = image.resize((target_width, target_height))
        logger.debug(f"✅没有进行裁剪图像原始尺寸: {image}")

    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # 创建5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)

def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    pad_height = target_height - source_height
    pad_width = target_width - source_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # 处理奇数填充
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # 处理奇数填充

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    logger.debug(f"✅图像填充: {padding}")
    return padding

def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    words = clean_text.split()

    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)

# 生成输出视频名称
def get_unique_filename(
    base: str,             # 基础文件名前缀
    ext: str,              # 文件扩展名(如".mp4")
    prompt: str,           # 原始提示词文本
    seed: int,
    resolution: tuple[int, int, int],  # 分辨率(宽,高,帧数)
    dir: Path,             # 文件存储目录路径
    endswith=None,         # 可选的文件名后缀
    index_range=1000,      # 尝试生成唯一文件名的最大次数 -> Path唯一的文件路径对象
) -> Path:
    # 将提示词转换为适合文件名的形式(截断/替换特殊字符等)
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):         # 通过添加序号确保文件名唯一
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"   # 构建完整文件名
        if not os.path.exists(filename):              # 如果文件不存在则返回该路径
            return filename
    # 如果循环结束仍未找到可用文件名则报错
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )

def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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

# 定义视频生成的管道，负责加载模型和组装主干 pipeline
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
    assert os.path.exists(ckpt_path), f"权重文件不存在: {ckpt_path}"

    # 自动选择 dtype
    dtype = get_supported_precision()
    total_gpu_mem = get_total_gpu_memory()
    is_low_mem_gpu = (total_gpu_mem > 0 and total_gpu_mem <= 8)

    # 使用device_map和low_cpu_mem_usage参数进行分片加载
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if is_low_mem_gpu:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": device or get_device()}

    with safe_open(ckpt_path, framework="pt") as f:
        metadata = f.metadata()
        config_str = metadata.get("config")
        configs = json.loads(config_str)
        logger.debug(f"✅主模型权重配置内容")
        allowed_inference_steps = configs.get("allowed_inference_steps", None)
        logger.debug(f"✅限制模型推理时可用的时间步: {allowed_inference_steps}")

    try:
        vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)                 # 加载自动编码器
        logger.debug(f"✅VAE模型加载成功: vae_path = {ckpt_path}")
    except Exception as e:
        logger.error(f"❌VAE模型加载失败：{ckpt_path}, error: {e}")
    try:
        transformer = Transformer3DModel.from_pretrained(ckpt_path)   
        logger.debug(f"✅Transformer模型加载成功:transformer_path = {ckpt_path}")
    except Exception as e:
        logger.error(f"❌Transformer模型加载失败：{ckpt_path}, error: {e}")

    # 如果采样器为"from_checkpoint"或没有sampler值，则执行if条件代码，从ckpt_path路径加载预训练模型；否则执行else分支代码。
    if sampler == "from_checkpoint" or not sampler:                                            
        scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)
        logger.debug(f"✅主模型采样器Scheduler加载成功: {scheduler}")
    else:
        scheduler = RectifiedFlowScheduler(
            sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
        )

    # 文本编碼器从T5模型加载
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_model_name_or_path, subfolder="text_encoder")
    logger.debug(f"✅文本编码模型加载成功: {text_encoder_model_name_or_path}, ")

    patchifier = SymmetricPatchifier(patch_size=1)      # 修补器

    tokenizer = T5Tokenizer.from_pretrained(text_encoder_model_name_or_path, subfolder="tokenizer")
    logger.debug(f"✅分词器加载成功: {text_encoder_model_name_or_path}")

    transformer = transformer.to(device)      # 将注意力机制模型transformer转移到设备上
    vae = vae.to(device)                      # 将自动编码器转移到设备上
    text_encoder = text_encoder.to(device)    # 将文本编码器转移到设备上

    if enhance_prompt:
        try:
            prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained( 
                prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
            )
            logger.debug(
                f"✅图像增强模型加载成功: {prompt_enhancer_image_caption_model_name_or_path},"
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
    logger.debug(f"✅传输模型配置字典给管道的视频处理流水线类")
    pipeline = pipeline.to(device)                    # 将管道转移到设备上
    return pipeline                                    # 返回管道

def create_latent_upsampler(latent_upsampler_model_path: str, device: str):
    # 创建空间上采样器（LatentUpsampler），它只在多尺度渲染（multi-scale pipeline）时才需要，并不是所有管道都用得上。
    latent_upsampler = LatentUpsampler.from_pretrained(latent_upsampler_model_path)
    latent_upsampler.to(device) 
    latent_upsampler.eval()
    logger.debug(f"✅潜空间上采样模型加载成功")
    logger.debug(f"✅潜空间上采样模型运行设备: {latent_upsampler.device}")
    return latent_upsampler

# 定义推断,负责调度流程、参数准备、调用 pipeline，以及后处理（如保存输出）
def infer(
    pipeline_config: str, 
    output_path: Optional[str] = None,                                        # 输出路径：自选
    seed: int = 42,                                                    # 种子
    image_cond_noise_scale: float = 0.0,                                     # 图像条件
    height: Optional[int] = None,                                            # 图像高：自选
    width: Optional[int] = None,
    num_frames: int = 120,
    frame_rate: float = 24,
    num_inference_steps: int = 20,
    stochastic_sampling: bool = False,
    decode_timestep: Union[List[float], float] = 0.0,
    decode_noise_scale: Optional[List[float]] = None,
    num_images_per_prompt: Optional[int] = 1,
    stg_rescale: float = 0.7,
    prompt: str = "",                                                    # 提示词
    negative_prompt: str = "",                                            # 反向提示词
    offload_to_cpu: bool = False,                                              # 将计算任务转移到cpu：Offload 技术将GPU显存中的权重卸载到CPU内存
    input_media_path: Optional[str] = None,                            # 输入媒体路径：自选列表
    conditioning_media_paths: Optional[List[str]] = None,              # 条件媒体路径：自选列表             
    conditioning_strengths: Optional[List[float]] = None,              # 条件强度：自选列表
    conditioning_start_frames: Optional[List[int]] = None,             # 条件开始框架：自选输入
    device: Optional[str] = None,                                      # 设备：自选
    **kwargs,                                                          # 关键字参数字典
):
    models_dir = "MODEL_DIR"

    ltxv_model_name_or_path = kwargs.get("checkpoint_path")
    ltxv_model_path = ltxv_model_name_or_path  
    logger.debug(f"✅LTX-Video模型存在: {ltxv_model_path}")
    spatial_upscaler_model_name_or_path = kwargs.get("spatial_upscaler_model_path")
    logger.debug(f"✅空间上采样模型路径存在: {spatial_upscaler_model_name_or_path}")
    # 获取媒体条件
    if kwargs.get("input_image_path", None):             # **kwargs代表关键字参数
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
    logger.debug(f"✅显存小于8GB设备运行在cpu: {offload_to_cpu}")

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
    prompt_enhancement_words_threshold = kwargs.get("prompt_enhancement_words_threshold")
    logger.debug(f"✅获取提示词阀值: {prompt_enhancement_words_threshold}")
    prompt_word_count = len(prompt.split())
    enhance_prompt = (
        prompt_enhancement_words_threshold > 0
        and prompt_word_count < prompt_enhancement_words_threshold
    )
    logger.debug(f"✅推理文件增强提示词: {enhance_prompt}")

    if prompt_enhancement_words_threshold > 0 and not enhance_prompt:
        logger.info(
            f"✅警告:提示词有{prompt_word_count}个, 这起过了配置文件中阀值{prompt_enhancement_words_threshold}.已禁用提示词增强."
        )

    precision = kwargs.get("precision")   # 获取精度
    logger.debug(f"✅获取精度: {precision}")

    text_encoder_model_name_or_path = kwargs.get("text_encoder_model_name_or_path")
    sampler = kwargs.get("sampler")    # 从data中获取采样器
    prompt_enhancer_image_caption_model_name_or_path = kwargs.get("prompt_enhancer_image_caption_model_name_or_path")
    prompt_enhancer_llm_model_name_or_path = kwargs.get("prompt_enhancer_llm_model_name_or_path")
    logger.debug(f"✅获取文本编码模型路径{text_encoder_model_name_or_path}, 获取采样器{sampler}, 获取图像增强{prompt_enhancer_image_caption_model_name_or_path}, 获取提示词增强模型{prompt_enhancer_llm_model_name_or_path}")

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

    # 从参数中获取管道类型,如果选择多尺度管道"multi-scale",
    pipeline_type = kwargs.get("pipeline_type")
    logger.debug(f"✅获取管道生成类型：{pipeline_type}")
    if pipeline_type == "multi-scale":
        spatial_upscaler_model_path = kwargs.get("spatial_upscaler_model_path")   # 获取空间上采样模型的路径,如果没有提供路径，则报错
        if not spatial_upscaler_model_path:
            raise ValueError(
                "spatial upscaler model path is missing from kwargs or pipeline config file and is required for multi-scale rendering"
            )
        latent_upsampler = create_latent_upsampler(spatial_upscaler_model_path, pipeline.device) # 创建潜在空间上采样器（latent upsampler）
        logger.debug(f"✅空间上采样模型存在：{spatial_upscaler_model_path}")
        first_pass = kwargs.get("first_pass") 
        second_pass = kwargs.get("second_pass")
        downscale_factor = kwargs.get("downscale_factor") 
        logger.debug(f"✅空间上采样获取参数：{first_pass}, {second_pass}, {downscale_factor}")

        pipeline = LTXMultiScalePipeline(
            pipeline,
            latent_upsampler=latent_upsampler,
            first_pass=first_pass,
            second_pass=second_pass,
            downscale_factor=downscale_factor,
        )
        logger.debug(f"✅传输管道配置给管道动态上采样类：{pipeline}")
    
    # 加载图像文件
    media_item = None
    if input_media_path:
        media_item = load_media_file(
            media_path=input_media_path,
            height=height,
            width=width,
            max_frames=num_frames_padded,
            padding=padding,
            device=device or get_device(),
        )

        # 添加检查点，如还出错则删除
        if media_item is None:
            logger.error(f"❌输入媒体文件{input_media_path}加载失败, 中止推理")
            raise RuntimeError(f"输入媒体文件{input_media_path}加载失败")

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
            device=device,
        )
        if conditioning_media_paths
        else None
    )

    # 添加检查点，如还出错则删除
    if conditioning_media_paths and conditioning_items is None:
        logger.error(f"❌所有条件媒体文件加载失败, 中止推理")
        raise RuntimeError("所有条件媒体文件加载失败")

    if conditioning_items:   
        for idx, item in enumerate(conditioning_items):  
            logger.debug(f"✅推理: 条件媒体 #{idx} device = {item.media_item.device}, target = {device}")

    # 注意力机制
    stg_mode = kwargs.get("stg_mode")
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
    logger.debug(f"✅时空引导模式注意力机制模式: {stg_mode}")

    # 为管道准备输入,构造 pipeline 输入的字典
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
    }
    device = device or get_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    logger.debug(f"✅管道获取设备: {device}")

    # 图像管道,实际推理调用
    images = pipeline(
        **pipeline_config,
        skip_layer_strategy=skip_layer_strategy,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,    # 管道回调函数，该回调函数在每一步结束时执行，并修改管道属性和变量，以供下一步使用
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=frame_rate,
        **sample,                                 # 提示词字典
        num_inference_steps=num_inference_steps,  # 新增必要参数
        stochastic_sampling=stochastic_sampling,  # 新增必要参数
        num_images_per_prompt=num_images_per_prompt,
        decode_timestep=decode_timestep,
        decode_noise_scale=decode_noise_scale,
        stg_rescale=stg_rescale,
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
                resolution=(height, width, num_frames),    # 分辨率（高，宽，帧率）
                dir=output_dir,
            )
            # 写入视频
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)
        logger.debug(f"✅输出文件名称: {output_filename}")
  
# 负责根据路径、强度、起始帧等，批量调用 ，并生成 ConditioningItem 列表供 pipeline 使用,负责“批量调度+组装”
def prepare_conditioning(
    conditioning_media_paths: List[str],
    conditioning_strengths: List[float],
    conditioning_start_frames: List[int],
    height: int,
    width: int,
    num_frames: int,
    padding: tuple[int, int, int, int],
    pipeline: LTXVideoPipeline,
    device:str
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
            device=device, 
        )

        # 增加检查点，定位错误，如还出错说明错误不在这里，则删除
        if media_tensor is None:
            logger.error(f"❌媒体文件{path}加载失败，跳过该条件")
            continue  # 跳过此项

        media_tensor = media_tensor.to(device)  # 新增：转到目标设备 
        conditioning_items.append(ConditioningItem(media_tensor, start_frame, strength))   # 媒体张量、开始帧、强度
        logger.debug(f"✅媒体张量已转移到设备: {media_tensor.device}") 

    # 增加检查点，定位错误，如还出错说明错误不在这里，则删除
    if not conditioning_items:
        logger.error(f"❌所有条件媒体文件加载失败，返回None")
        return None

    logger.debug(f"✅获取条件参数: {conditioning_items}")
    return conditioning_items

# 获取媒体帧参数,用于判断输入媒体的帧数（图片为 1，视频为实际帧数）负责“单个媒体帧数的获取”
def get_media_num_frames(media_path: str) -> int:
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]
    )
    num_frames = 1
    if is_video:
        reader = imageio.get_reader(media_path)
        num_frames = reader.count_frames()
        reader.close()
        logger.debug(f"✅媒体帧参数: {num_frames}")
    return num_frames

# 加载媒体文件,负责“单个媒体文件的读取和预处理”
def load_media_file(
    media_path: str,
    height: int,
    width: int,
    max_frames: int,
    padding: tuple[int, int, int, int],
    just_crop: bool = False,
    device: str = "cpu",   # 新增参数，默认cpu
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
                frame, height, width, just_crop=just_crop, device=device
            )

            # 增加检查，如还出错则不是这里问题，应删除
            if frame_tensor is None:
                logger.error(f"❌帧{media_path}第{i}帧加载失败, 跳过")
                continue  # 跳过该帧

            frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
            frames.append(frame_tensor)
            logger.debug(f"✅媒体帧预处理张量运行在: {frame_tensor.device}") 
        reader.close()

        # 增加检查，如还出错则不是这里问题，应删除
        if not frames:
            logger.error(f"❌视频{media_path}全部帧加载失败")
            return None

        media_tensor = torch.cat(frames, dim=2)
    else:                                                                            # 输入图像
        media_tensor = load_image_to_tensor_with_resize_and_crop(
            media_path, height, width, just_crop=just_crop, device=device
        )

        # 增加检查，如还出错则不是这里问题，应删除
        if media_tensor is None:
            logger.error(f"❌图像{media_path}加载失败")
            return None

        logger.debug(f"✅media_tensor(load img) device: {media_tensor.device}")

        media_tensor = torch.nn.functional.pad(media_tensor, padding)
        logger.debug(f"✅media_tensor(after pad) device: {media_tensor.device}")

        media_tensor = media_tensor.to(device)
        logger.debug(f"✅媒体张量运行在: {media_tensor.device }")
        logger.debug(f"✅media_tensor(final return) device: {media_tensor.device}")
    return media_tensor

if __name__ == "__main__":
    main()
