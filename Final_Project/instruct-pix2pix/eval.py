import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchmetrics import SSIM
from torchmetrics.image import PSNR
from torchvision.transforms import ToTensor
from tqdm import tqdm
from omegaconf import OmegaConf

import sys
sys.path.append("./stable_diffusion")

# 新增导入
import k_diffusion as K
import einops
from einops import rearrange
from ldm.util import instantiate_from_config

# 新增 CFGDenoiser 类定义
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


# -------------------------- 核心模型类定义 --------------------------
class InstructPix2Pix(nn.Module):
    def __init__(self, config_path: str, checkpoint_path: str, vae_ckpt: str = None):
        super().__init__()
        # 加载配置文件
        self.config = OmegaConf.load(config_path)
        
        # 实例化模型架构
        self.model = instantiate_from_config(self.config.model)
        
        # 加载模型权重
        pl_sd = torch.load(checkpoint_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        
        # 加载VAE权重（如果有）
        if vae_ckpt is not None:
            vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
            sd = {
                k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
                for k, v in sd.items()
            }
        
        self.model.load_state_dict(sd, strict=False)
        self.model.to("cuda")  # 显式转移到 GPU
        self.model.eval()
        
        # 初始化扩散参数
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""]).to("cuda") 

    @classmethod
    def load_from_checkpoint(cls, config_path: str, checkpoint_path: str, vae_ckpt: str = None):
        return cls(config_path, checkpoint_path, vae_ckpt)
    
    def edit(
        self,
        image: torch.Tensor,
        instruction: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> torch.Tensor:
        """生成编辑后的图像（与原作代码对齐）"""
        image = image.to("cuda")
        with torch.no_grad(), self.model.ema_scope():
            # 构建条件输入
            cond = {
                "c_crossattn": [self.model.get_learned_conditioning([instruction]).to("cuda")],
                "c_concat": [self.model.encode_first_stage(image.unsqueeze(0)).mode().to("cuda")],
            }
            uncond = {
                "c_crossattn": [self.null_token.to("cuda")],
                "c_concat": [torch.zeros_like(cond["c_concat"][0]).to("cuda")],
            }
            
            # 扩散采样
            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": guidance_scale,
                "image_cfg_scale": 1.0,
            }
            sigmas = self.model_wrap.get_sigmas(num_inference_steps)
            x = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            x = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args)
            
            # 解码生成图像
            return self.model.decode_first_stage(x)[0]

# -------------------------- 工具函数 --------------------------
# 在数据预处理时强制统一尺寸
def preprocess_image(image: Image.Image, target_size: tuple = (256, 256)) -> Image.Image:
    return image.resize(target_size, Image.Resampling.LANCZOS)

def load_model(
    config_path: str,
    checkpoint_path: str,
    vae_ckpt: str = None,
    device: str = "cuda"
) -> InstructPix2Pix:
    """加载模型实例"""
    model = InstructPix2Pix.load_from_checkpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        vae_ckpt=vae_ckpt
    )
    return model.to(device)

def generate_prediction(
    model: InstructPix2Pix,
    input_image: Image.Image,
    instruction: str,
    device: str = "cuda",
    steps: int = 50,
    guidance_scale: float = 7.5
) -> Image.Image:
    """生成预测图像（兼容原作参数）"""
    # 图像预处理
    input_tensor = ToTensor()(input_image).to(device)
    
    # 生成预测
    output_tensor = model.edit(
        input_tensor,
        instruction=instruction,
        num_inference_steps=steps,
        guidance_scale=guidance_scale
    )
    
    # 后处理：Tensor → PIL Image
    output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_image = (output_image * 255).astype(np.uint8)
    return Image.fromarray(output_image)

def evaluate_metrics(pred_image: Image.Image, target_image: Image.Image, device: str = "cuda") -> dict:
    """计算SSIM和PSNR"""
    ssim = SSIM(data_range=255.0).to(device)
    psnr = PSNR(data_range=255.0).to(device)
    
    pred_tensor = ToTensor()(pred_image).unsqueeze(0).to(device)
    target_tensor = ToTensor()(target_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ssim_score = ssim(pred_tensor, target_tensor)
        psnr_score = psnr(pred_tensor, target_tensor)
    
    return {"SSIM": ssim_score.item(), "PSNR": psnr_score.item()}

# -------------------------- 主函数 --------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    print("Loading model...")
    model = load_model(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        vae_ckpt=args.vae_ckpt,
        device=device
    )
    
    # 创建输出目录
    pred_dir = Path(args.output_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历数据集
    data_root = Path(args.data_dir)
    all_metrics = []
    
    for sample_dir in tqdm(list(data_root.glob("sample_*")), desc="Processing samples"):
        # 读取输入数据
        input_path = next(sample_dir.glob("*_0.jpg"))
        target_path = next(sample_dir.glob("*_1.jpg"))
        with open(sample_dir / "prompt.json") as f:
            instruction = json.load(f)["edit"]
        
        input_image = preprocess_image(Image.open(input_path))
        target_image = preprocess_image(Image.open(target_path))
        
        # 生成预测
        pred_image = generate_prediction(
            model, input_image, instruction, 
            steps=args.steps, guidance_scale=args.guidance_scale
        )
        
        # 保存结果
        output_path = pred_dir / f"{sample_dir.name}_pred.jpg"
        pred_image.save(output_path)
        
        # 计算指标
        metrics = evaluate_metrics(pred_image, target_image, device)
        all_metrics.append(metrics)
    
    # 汇总结果
    avg_ssim = np.mean([m["SSIM"] for m in all_metrics])
    avg_psnr = np.mean([m["PSNR"] for m in all_metrics])
    
    print("\nEvaluation Results:")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")


data_dir = "./data/instructpix2pix"
checkpoint_path = "./logs/train_default/checkpoints/last.ckpt"
config_path = "./configs/train.yaml"
output_dir = "./logs/predictions"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必要参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据集根目录（包含 sample_* 子目录）")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点路径（.ckpt 文件）")
    parser.add_argument("--config_path", type=str, required=True, help="配置文件路径（如 configs/train.yaml）")
    
    # 可选参数
    parser.add_argument("--output_dir", type=str, default="predictions", help="预测图像输出目录")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="VAE 检查点路径（如有）")
    parser.add_argument("--steps", type=int, default=50, help="扩散采样步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="文本引导比例")
    
    args = parser.parse_args()
    main(args)