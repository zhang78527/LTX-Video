# 提示增强工具
import logging
from typing import Union, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


T2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the image first and then add the user input. Image description should be in first priority! Align to the image caption if it contradicts the user text input.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Align to the image caption if it contradicts the user text input.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

# 将张量转换为PIL图像：tensor：PyTorch 张量，形状为 (C, H, W)（通道数, 高度, 宽度）
def tensor_to_pil(tensor):

    # 确保张量在 [-1, 1]范围内
    assert tensor.min() >= -1 and tensor.max() <= 1

    # 值范围转换从 [-1, 1] 转换为 [0, 1]
    tensor = (tensor + 1) / 2

    # 调整张量维度顺序从 [C, H, W] 重排为 [H, W, C]
    tensor = tensor.permute(1, 2, 0)

    # 先转换为 numpy 数组，再转换8位无符号整数（uint8） 范围 [0, 255]
    numpy_image = (tensor.cpu().numpy() * 255).astype("uint8")

    # 转换为 PIL 图像
    return Image.fromarray(numpy_image)

# 生成电影风格文本提示（prompt） 的工具，用于视频生成或图像到视频转换任务
def generate_cinematic_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompt: Union[str, List[str]],
    conditioning_items: Optional[List] = None,
    max_new_tokens: int = 256,
) -> List[str]:
    # 将单个提示转换为列表形式，统一处理格式
    prompts = [prompt] if isinstance(prompt, str) else prompt

    # 无条件生成（纯文本到视频）
    if conditioning_items is None:
        # 调用 _generate_t2v_prompt（文本到视频提示生成）
        prompts = _generate_t2v_prompt(
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            max_new_tokens,
            T2V_CINEMATIC_PROMPT,
        )
    else:
        # 有条件生成（图像到视频）:条件检查
        if len(conditioning_items) > 1 or conditioning_items[0].media_frame_number != 0:
            logger.warning(
                "prompt enhancement does only support unconditional or first frame of conditioning items, returning original prompts"
            )
            return prompts

        # 提取第一帧
        first_frame_conditioning_item = conditioning_items[0]
        first_frames = _get_first_frames_from_conditioning_item(
            first_frame_conditioning_item
        )

        # 数量匹配检查
        assert len(first_frames) == len(
            prompts
        ), "Number of conditioning frames must match number of prompts"

        # 图像到视频提示生成
        prompts = _generate_i2v_prompt(
            image_caption_model,
            image_caption_processor,
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            first_frames,
            max_new_tokens,
            I2V_CINEMATIC_PROMPT,
        )
    logger.debug(f"✅电影风格提示词生成: {prompts}")

    return prompts

# 从条件项中提取第一帧图像
def _get_first_frames_from_conditioning_item(conditioning_item) -> List[Image.Image]:
    # 获取视频帧张量：
    frames_tensor = conditioning_item.media_item
    return [
        tensor_to_pil(frames_tensor[i, :, 0, :, :])
        for i in range(frames_tensor.shape[0])
    ]

# 通过一个文本增强模型（例如大型语言模型）来增强文本提示
def _generate_t2v_prompt(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    system_prompt: str,
) -> List[str]:
    # 构建消息结构
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user_prompt: {p}"},
        ]
        for p in prompts
    ]

    # 将对话结构转换为模型接受的文本格式
    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    # 准备模型输入:将文本转换为token ID张量
    model_inputs = prompt_enhancer_tokenizer(texts, return_tensors="pt").to(
        prompt_enhancer_model.device
    )
    # 生成并解码提示:输出为文本字符串
    return _generate_and_decode_prompts(
        prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens
    )
    logger.debug(f"✅无条件生成视频解码后的提示词: {generate_and_decode_prompts}")

def _generate_i2v_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    first_frames: List[Image.Image],
    max_new_tokens: int,
    system_prompt: str,
) -> List[str]:
    image_captions = _generate_image_captions(
        image_caption_model, image_caption_processor, first_frames
    )

    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user_prompt: {p}\nimage_caption: {c}"},
        ]
        for p, c in zip(prompts, image_captions)
    ]

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    model_inputs = prompt_enhancer_tokenizer(texts, return_tensors="pt").to(
        prompt_enhancer_model.device
    )

    return _generate_and_decode_prompts(
        prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens
    )
    logger.debug(f"✅有条件生成视频解码后的提示词: {generate_and_decode_prompts}")

def _generate_image_captions(
    image_caption_model,
    image_caption_processor,
    images: List[Image.Image],
    system_prompt: str = "<DETAILED_CAPTION>",
) -> List[str]:
    image_caption_prompts = [system_prompt] * len(images)
    inputs = image_caption_processor(
        image_caption_prompts, images, return_tensors="pt"
    ).to(image_caption_model.device)

    with torch.inference_mode():
        generated_ids = image_caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    return image_caption_processor.batch_decode(generated_ids, skip_special_tokens=True)


def _generate_and_decode_prompts(
    prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens: int
) -> List[str]:
    with torch.inference_mode():
        outputs = prompt_enhancer_model.generate(
            **model_inputs, max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        ]
        decoded_prompts = prompt_enhancer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
    logger.debug(f"✅提示词增强填充: {decoded_prompts}")

    return decoded_prompts
