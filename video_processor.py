import os
import re
import sys
import cv2
import torch
import json
import asyncio
import time 
import yaml
import glob
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Callable
from safetensors import safe_open
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "models/LTX-Video"))
from safetensors.torch import load_file
from inference import infer, create_ltx_video_pipeline
from inference import InferenceModule
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

from src.gui.data_manager import DataManager, MessageTypes

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, config,data_manager, project_root, device):
        # 初始化核心参数
        self.config = config
        self.data_manager = None
        self.device = device

        self.material_paths = []
        self.pipeline = None
        self.current_command = None
        self._init_paths()
        self._setup_data_manager()
        self._load_models()
        # 初始化子模块
        self._init_inference_module()
        logger.info(f"✅文生视频初始化完成")
      
    def _init_paths(self):
        """初始化路径"""
        project_root = Path(__file__).parent.parent.parent.resolve()
        self.project_root = project_root
        self.output_dir = self.project_root / "output/video_processor"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Output directory set to: %s", self.output_dir)
        logger.info(f"文生视频输出目录初始化完成: {self.output_dir}")

        #模型存储路径
        self.checkpoint_path = self.project_root / "models" / "ltx-video-2b-v0.9.5.safetensors"
        self.text_encoder_model = self.project_root / "models" / "t5"
        self.image_caption_model = self.project_root / "models" / "Florence-2-base-PromptGen-v2.0" 
        self.prompt_enhancement_words = self.project_root / "models" / "Llama-3.2-1B-Instruct" / "intree"
        self.zh_en = self.project_root / "models" / "opus-mt-zh-en"
        self.spatial_upscaler_model_path = self.project_root / "models" / "ltxv-spatial-upscaler-0.9.7.safetensors"
        self.pretrained_model_path = self.project_root / "models" / "LTX-Video" / "ltx_video" / "models"
        logger.debug(f"文生视频权重文件存在: {self.checkpoint_path.exists()}") 
        logger.debug(f"文本编码模型存在: {self.text_encoder_model.exists()}")
        logger.debug(f"图像转文本模型存在: {self.image_caption_model.exists()}")
        logger.debug(f"提示词增强模型存在: {self.prompt_enhancement_words.exists()}")
        logger.debug(f"中英文翻译模型存在: {self.zh_en.exists()}")
        logger.debug(f"3D模型存在: {self.pretrained_model_path.exists()}")

        self.inference_path = self.project_root / "models " / "LTX-Video"/ "inference.py"                            
        self.pipeline_type = self.project_root / "models" / "LTX-Video" / "ltx_video\pipelines"/ "pipeline_ltx_video.py" 
       
    def _setup_data_manager(self):
        """配置数据管理器"""
        self.data_manager = DataManager()
        self.data_manager.register_event(MessageTypes.GENERATION_DATA, self._generation_data)
        logger.info("文生视频模块事件监听器已设置")

    def _load_models(self):
        """加载中文到英文模型"""
        model_name = self.zh_en
        try:
            # 优先 Marian
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            logger.debug(f"✅中英翻译模型加载成功")
            logger.debug(f"✅中英翻译模型分词器加载成功")

        except Exception as e:
            logger.error(f"MarianMT加载失败，尝试AutoTokenizer/AutoModel: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            logger.info("加载AutoTokenizer/AutoModel成功")
        
    def _generation_data(self, data):
        """接收执行指令任务"""
        if data.get("source") != "generation_window":
            return

        if data.get("action") != "execute":
            return

        # 3. 提取核心任务数据
        task_data = data.get("data", {})
        logger.debug(f"✅文生视频接收执行数据成功: {task_data}")
        if not task_data:
            logger.error("任务数据为空")
            return

        module_name = task_data.get("module_name", "")
        if module_name != "video_processor":
            return 

        action_type = task_data["action"] 
        parameters = task_data.get("parameters", {})                        # 获取参数
        materials = task_data.get("materials", [])
        original_command = task_data.get("original_command", "")            
        logger.info(
            f"视频生成模块收到生成任务: action={action_type}, "
            f"参数={parameters}, 素材={materials}, 提示词={original_command} "

        )

        self._execute_generation({
            "module_name": module_name,
            "action": action_type,
            "parameters": parameters,
            "materials": materials,
            "original_command": original_command
        })

    def _init_inference_module(self):
        """初始化推理模块"""
        try:
            self.inference_module = InferenceModule(
                config=self.config,
                data_manager=self.data_manager,
                project_root=self.project_root,
                device=self.device
            )
            logger.info(f"✅推理模块初始化成功")
        except Exception as e:
            logger.error(f"推理模块初始化失败: {str(e)}")
            self.inference_module = None

    def _execute_generation(self, task_data):
        """执行生成任务"""
        try:
            # 发送进度给文件生成窗口激活进度条

            if not task_data:
                raise ValueError("未接收到有效指令")

            # 传递素材参数原始提示词给视频生成逻辑
            action = task_data.get("action")
            result = None
            if action == "video_generation":
                result = self._process_video_generation(
                    task_data.get("materials", []),
                    task_data.get("parameters", {}),
                    task_data.get("original_command", "")
                )

            # 发送进度更新
       
            # 发送生成任务结果及文件存储路径给生成窗口
            self.data_manager.send_message(MessageTypes.OUTPUT_PATH, {
                "source": "video_processor",
                "status": "ready",
                "message": "生成任务完成",
                "output_paths": result
            })

        except Exception as e:
            logger.error(f"执行生成任务失败: {str(e)}")
            self.data_manager.send_message(MessageTypes.TASK_RESULT, {
                "module": "video_processor",
                "status": "error",
                "error": str(e)
            })

    def _process_video_generation(self, materials, command, original_command):
        """处理视频生成的核心逻辑"""
        try:
            patch_size = 32

            original_prompt = original_command            # 获取原始提示词
            logger.debug(f"✅获取原始提示词成功: {original_prompt}")

            # 判断 original_prompt 是否为中文
            def contains_chinese(text):
                for ch in text:
                    if '\u4e00' <= ch <= '\u9fff':
                        return True
                return False

            if contains_chinese(original_prompt):
                # 将文本转换为模型输入格式
                inputs = self.tokenizer(original_prompt, return_tensors="pt", padding=True, truncation=True)
                # 生成翻译
                translated = self.model.generate(**inputs)
                # 将生成的 ID 序列转换为文本
                prompt = self.tokenizer.decode(translated[0], skip_special_tokens=True)
                logger.debug(f"✅获取翻译后的提示词成功: {prompt}")
            else:
                prompt = original_prompt
                logger.debug("✅原始提示词为英文，无需翻译")

            # 卸载所有模型
            self.maybe_free_model_hooks()

            # 获取时长、帧率、分辨率
            default_frame_rate = 24
            frame_rate = int(command.get("frame_rate", default_frame_rate))
            duration_sec = float(command.get("duration", 5))
            num_frames = int(duration_sec * frame_rate)
            resolution_str = str(command.get("resolution", '720P(1280*720)') or '720P(1280*720)')
            match = re.search(r'(\d+)\s*\D?\s*(\d+)', resolution_str)
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
            else:
                width, height = 1280, 720
            aspect_ratio = command.get('aspect_ratio', '16:9')

            negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

            # 强制8G显卡参数限制
            gpu_mem_gb = 0
            try:
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    torch.cuda.empty_cache()
            except Exception:
                gpu_mem_gb = 0
            if gpu_mem_gb > 0 and gpu_mem_gb <= 8:
                height = min(height, 720)
                width = min(width, 1280)
                num_frames = min(num_frames, 64)
                logger.info(f"检测到8G显卡，降级分辨率/帧数：height={height},width={width},num_frames={num_frames}")

            # 统一padded尺寸，保证为patch_size整数倍
            height_padded = ((height - 1) // patch_size + 1) * patch_size
            width_padded = ((width - 1) // patch_size + 1) * patch_size

            # num_frames 按原有逻辑补齐
            num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
            device=self.device
            logger.info(f"✅ 视频生成模块获取设备: {device}")

            # 1.组装 params
            params = {
                "output_path": str(self.output_dir),                   # 文件输出路径
                "height": height_padded,
                "width":  width_padded,
                "num_frames": num_frames_padded,                       # 视频总帧数（若生成视频）从数据管理器传递过来的参数中自动获取
                "frame_rate": frame_rate,                              #  视频帧率帧数为8的倍数加1，自动从数据管理器传递过来的参数中获取
                "prompt": prompt,                                      # 提示词
                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                "pipeline_type": "multi-scale",          # "multi-scale", "base"
                "seed": 42,                                           # 种子
                "num_images_per_prompt": 1,       # 每个提示词生成的图像/视频数量（此处为 1 个，可能用于测试或单样本生成）
                "num_inference_steps": 15,        # 去噪步数,值越大生成质量可能越高,但耗时增加,40-50 次去噪步骤,足以得到一个高质量图像
                "stg_mode": "transformer_block",   # 模式 注意力机制选项（"attention_values", "attention_skip", "residual", "transformer_block"）
                "decode_timestep": 0.03,          # 解码过程的时间步长（控制生成过程的细粒度，值越小生成越精细但耗时增加）
                "decode_noise_scale": [0.015],      # 解码阶段的噪声强度（影响生成结果的多样性与清晰度）
                "stg_rescale": 0.7, 
                "image_cond_noise_scale": 0.10,                           # 条件图像的噪声缩放因子，控制条件图像对生成结果的影响强度。
                "prompt_enhancement_words_threshold": 120,                # 提示词限制
                "offload_to_cpu": (gpu_mem_gb > 0 and gpu_mem_gb <= 8),   # 8G显卡强制CPU卸载
                "precision": "bfloat16",
                "device": device,
                "downscale_factor": 0.6666666,
                "stochastic_sampling": False,   # 启用随机采样：True生成结果随机性，但不可复现，False生成结果确定，适合测试可重复性
            }
            # 2.组装配置
            config = {
                "checkpoint_path": str(self.checkpoint_path),                                       # 主模型权重路径
                "pretrained_model_path": str(self.pretrained_model_path),
                "text_encoder_model_name_or_path": str(self.text_encoder_model),                    # 文本编码模型路径
                "prompt_enhancer_image_caption_model_name_or_path": str(self.image_caption_model),  # 图像增强模型路径
                "prompt_enhancer_llm_model_name_or_path": str(self.prompt_enhancement_words),       # 提示词增强模型路径
                "spatial_upscaler_model_path": str(self.spatial_upscaler_model_path),               # 空间采样模型路径
                "sampler": "uniformt",        # 选项: "uniform", "linear-quadratic", "from_checkpoint"
            }       # "uniform"：均匀采样器;"linear-quadratic"：使用线性二次采样器;"from_checkpoint"：从检查点加载调度器配置，而不是重新创建

            # 3.组装 conditioning_params
            conditioning_params = {}
            if materials:
                # 只用第一个素材作为输入（如流程有多素材合成可扩展此处）
                input_media_path = materials[0]
                conditioning_params["conditioning_media_paths"] = [input_media_path]
                conditioning_params["conditioning_start_frames"] = [0]
                conditioning_params["conditioning_strengths"] = [1.0]
            # 多阶段的生成过程，参数随时间变化
            first_pass = {
                "guidance_scale": [1, 1, 6, 8, 6, 1, 1],  # 控制文本提示词对生成结果的影响强度,值越大图像质量越好,3-3.5
                "stg_scale": [0, 0, 4, 4, 4, 2, 1],       # 提示词引导系数。参数并不是越大越好，值为3时，与提示词相近，合理的值（3-5）
                "rescaling_scale": [1, 1, 0.5, 0.5, 1, 1, 1],# 缩放0.5-0.7
                "guidance_timesteps": [1.0, 0.996,  0.9933, 0.9850, 0.9767, 0.9008, 0.6180],# 时间步通常是从噪声水平高到低（从1到0）。
                "skip_block_list": [[], [11, 25], [22], [27], [27], [27], [27]], # 指定在每个阶段跳过哪些网络块
                "skip_final_inference_steps": 0,      # 跳过最后的推理步骤
                "cfg_star_rescale": True,             # 是否使用CFG*重缩放技术
            }
            # 单阶段的精炼过程，使用固定的参数
            second_pass = {
                "guidance_scale": [4],              # 控制文本提示词对生成结果的影响强度,值越大图像质量越好,3-3.5
                "stg_scale": [4],                   # 提示词引导系数。参数并不是越大越好，值为3时，与提示词相近，合理的值（3-5）
                "rescaling_scale": [1],           # 缩放0.5-0.7
                "guidance_timesteps": [1.0],        # 时间步通常是从噪声水平高到低（从1到0）。
                "skip_block_list": [0],             # 指定在每个阶段跳过哪些网络块
                "skip_initial_inference_steps": 0,  # 跳过初始的推理步骤
                "cfg_star_rescale": True,           # 是否使用CFG*重缩放技术
            }

            # 4.组装inference_data
            inference_data = {
                "params": params,
                "config": config,
                "conditioning_params": conditioning_params,
                "first_pass": first_pass,
                "second_pass": second_pass,
            }

            self.data_manager.send_message(           
                MessageTypes.INFERENCE_DATA,             
                {
                    "source": "generate_video",  # 来源标识
                    "data": inference_data
                }
            )
            logger.info(f"✅ 发送推理参数至推理模块: {inference_data}")

            video_files = glob.glob(str(self.output_dir / "video_output_*.mp4"))
            if not video_files:
                raise RuntimeError("未找到生成的视频文件")
            latest_video = max(video_files, key=os.path.getctime)
            return [latest_video]
        except Exception as e:
            logger.error(f"视频生成任务失败: {str(e)}")
            raise

    def _save_video(self, frames: List[Image.Image], fps: int) -> Path:
        """保存结果"""
        output_path = self.output_dir / f"video_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (frames[0].width, frames[0].height)
        )   

        try:
            for frame in frames:
                cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                writer.write(cv_frame)
            return output_path
        finally:
            writer.release()
            logger.info(f"视频保存成功: {output_path}")

    def maybe_free_model_hooks(self):
        """卸载已加载的翻译模型，释放显存"""
        try:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()  # 如果在GPU上
            logger.info("已卸载中英翻译模型并清理CUDA缓存")
        except Exception as e:
            logger.warning(f"卸载模型时发生异常: {e}")

    def _update_status(self, value: int, message: str):
        """统一更新进度"""
        logger.info(f"状态更新: {message}")        
        self.data_manager.send_message(MessageTypes.PROCESS_PROGRESS, {
            "progress": value,
            "message": f"VideoProcessor - {message}",
            "module": "video_processor"
        })
