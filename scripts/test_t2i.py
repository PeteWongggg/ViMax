from diffusers import DiffusionPipeline
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # 指定第0和第1张卡

model_name = "/nas/models/t2i/qwen-image-2512/Qwen/Qwen-Image-2512"

pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="balanced"   # 自动把模型权重均匀分到两张卡
)

# 不需要手动 .to(device)，device_map 已经处理了

prompt = "一个成年人"
negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

width, height = 1664, 928

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda:0").manual_seed(42)
).images[0]

image.save("example.png")