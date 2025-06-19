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
        logger.info("文生视频初始化完成")
      
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
        self.spatial_upscaler_model_path = self.project_root / "models" / "ltx-video-2b-v0.9.5.safetensors"

        self.inference_path = self.project_root / "models " / "LTX-Video"/ "inference.py"                            
        self.pipeline_type = self.project_root / "models" / "LTX-Video" / "ltx_video\pipelines"/ "pipeline_ltx_video.py" 
        self.pipeline_config = self.project_root / "models" / "LTX-Video" / "configs" / "ltxv-2b-0.9.5.yaml"
        if not os.path.isfile(self.pipeline_config):
            raise RuntimeError(f"pipeline_config 配置文件不存在: {self.pipeline_config}")        
        logger.debug(f"文生视频权重文件存在: {self.checkpoint_path.exists()}") 
        logger.debug(f"文本编码模型存在: {self.text_encoder_model.exists()}")
        logger.debug(f"图像转文本模型存在: {self.image_caption_model.exists()}")
        logger.debug(f"提示词增强模型存在: {self.prompt_enhancement_words.exists()}")
        logger.debug(f"中英文翻译模型存在: {self.zh_en.exists()}")

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

    def _execute_generation(self, task_data):
        """执行生成任务"""
        try:
            # 发送进度给文件生成窗口激活进度条
            self._update_status(20, "开始执行生成任务")
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
            self._update_status(80, "生成任务即将完成")
            self._update_status(90, "生成任务完成")
       
            # 发送生成任务结果及文件存储路径给生成窗口
            self.data_manager.send_message(MessageTypes.OUTPUT_PATH, {
                "source": "video_processor",
                "status": "ready",
                "message": "生成任务完成",
                "output_paths": result
            })

            self._update_status(100, "生成任务完成")
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
            self._update_status(10, "开始处理视频生成请求")
            
            patch_size = 32

            original_prompt = original_command            # 获取原始提示词
            logger.debug(f"✅获取原始提示词成功: {original_prompt}")

            # 将文本转换为模型输入格式
            inputs = self.tokenizer(original_prompt, return_tensors="pt", padding=True, truncation=True)

            # 生成翻译
            translated = self.model.generate(**inputs)

            # 将生成的 ID 序列转换为文本
            prompt = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            logger.debug(f"✅获取翻译后的提示词成功: {prompt}")

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

      
            # 4. 统一padded尺寸，保证为patch_size整数倍
            height_padded = ((height - 1) // patch_size + 1) * patch_size
            width_padded = ((width - 1) // patch_size + 1) * patch_size

            # 5. num_frames 按原有逻辑补齐
            num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

            # 2. 组装 params
            params = {
                "output_path": str(self.output_dir),
                "seed": 42,                                           # 种子
                "pipeline_config": str(self.pipeline_config),
                "num_inference_steps": 40,                            # 去噪步数,值越大生成质量可能越高,但耗时增加,50 次去噪步骤,足以得到一个高质量图像
                "image_cond_noise_scale": 0.10,                       # 条件图像的噪声缩放因子，控制条件图像对生成结果的影响强度。
                "height": height_padded,
                "width":  width_padded,
                "num_frames": num_frames_padded,                       # 视频总帧数（若生成视频）从数据管理器传递过来的参数中自动获取
                "frame_rate": frame_rate,                              #  视频帧率帧数为8的倍数加1，自动从数据管理器传递过来的参数中获取
                "prompt": prompt,
                "negative_prompt": "distorted, deformed, blurry, low quality, mutated hands, extra limbs, disfigured, worst quality, jittery, flickering, inconsistent motion, watermark, text",
                "offload_to_cpu": (gpu_mem_gb > 0 and gpu_mem_gb <= 8), # 8G显卡强制CPU卸载
                "negative_prompt": negative_prompt

            }

            config = {
                "pipeline_type": "base",
                "num_images_per_prompt": 1,       # 每个提示词生成的图像/视频数量（此处为 1 个，可能用于测试或单样本生成）
                "guidance_scale": 8,            # 控制文本提示词对生成结果的影响强度,值越大图像质量越好,7和8.5之间通常是稳定扩散的好选择
                "stg_scale": 8,                   # 提示词引导系数。参数并不是越大越好，值为3时，与提示词相近，合理的值（7-10）
                "stg_rescale": 0.7,               # 缩放
                "stg_mode": "attention_values",   # 模式 注意力机制
                "stg_skip_layers": "1,2,3",       # 跳过某些网络层的索引（如跳过第 1、2、3 层，可能用于简化计算或定制生成效果）
                "precision": "float16",
                "decode_timestep": 0.03,          # 解码过程的时间步长（控制生成过程的细粒度，值越小生成越精细但耗时增加）
                "decode_noise_scale": 0.015,      # 解码阶段的噪声强度（影响生成结果的多样性与清晰度）。
                "device": self.device,
                "pipeline_config": str(self.pipeline_config.resolve()),                                       # 主模型配置文件路径
                "checkpoint_path": str(self.checkpoint_path.resolve()),                                       # 主模型权重路径
                "text_encoder_model_name_or_path": str(self.text_encoder_model.resolve()),                    # 文本编码模型路径
                "prompt_enhancer_image_caption_model_name_or_path": str(self.image_caption_model.resolve()),  # 图像增强模型路径
                "prompt_enhancer_llm_model_name_or_path": str(self.prompt_enhancement_words.resolve()),       # 提示词增强模型路径
                "spatial_upscaler_model_path": str(self.spatial_upscaler_model_path.resolve()),
                "prompt_enhancement_words_threshold": 120,                                                    # 提示词限制
                "stochastic_sampling": False,
                "sampler": "from_checkpoint"
            }

            # 4. 组装 conditioning_params
            conditioning_params = {}
            if materials:
                # 只用第一个素材作为输入（如流程有多素材合成可扩展此处）
                input_media_path = materials[0]
                conditioning_params["conditioning_media_paths"] = [input_media_path]
                conditioning_params["conditioning_start_frames"] = [0]
                conditioning_params["conditioning_strengths"] = [1.0]

            # 5. 合并所有参数并推理
            all_params = {**conditioning_params, **params, **config}
            # 关键：强制开启增强
            all_params['enhance_prompt'] = True

            # 8G显卡强制CPU卸载
            if gpu_mem_gb > 0 and gpu_mem_gb <= 8:
                all_params['offload_to_cpu'] = True

            self._update_status(20, f"参数准备完成，开始合成(height={512},width={768},num_frames={num_frames},offload={all_params['offload_to_cpu']})")

            infer(**all_params)
            self._update_status(100, "视频生成完成")

            video_files = glob.glob(str(self.output_dir / "video_output_*.mp4"))
            if not video_files:
                raise RuntimeError("未找到生成的视频文件")
            latest_video = max(video_files, key=os.path.getctime)
            return [latest_video]

        except Exception as e:
            logger.error(f"视频生成任务失败: {str(e)}")
            self._update_status(-1, f"视频生成任务失败: {str(e)}")
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
