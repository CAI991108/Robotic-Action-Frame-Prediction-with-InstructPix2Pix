#%%
import re
import json
from pathlib import Path

def generate_seeds_json(dataset_root):
    """
    生成与样本目录对应的seeds.json
    修改说明：
    1. 使用目录序号作为确定性的6位种子（前导零补齐）
    2. 保持目录顺序的数字连续性
    :param dataset_root: 数据集根目录 (包含sample_* 目录的路径)
    """
    dataset_path = Path(dataset_root)
    
    # 扫描并排序样本目录
    samples = []
    for dir_path in dataset_path.iterdir():
        if dir_path.is_dir() and (match := re.match(r"sample_(\d+)", dir_path.name)):
            samples.append((int(match.group(1)), dir_path.name))  # 保存序号和目录名
    samples.sort(key=lambda x: x[0])  # 按照序号排序
    
    # 生成确定性的种子列表
    seeds = [{
        "name": dir_name,       # 保持原目录名格式
        "seed": f"{i:06d}"              # 6位前导零格式
    } for i, dir_name in samples]
    
    # 写入JSON文件
    output_path = dataset_path / "seeds.json"
    with open(output_path, "w") as f:
        json.dump(seeds, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 配置路径（与之前脚本的输出目录一致）
    dataset_root = "./"
    
    generate_seeds_json(dataset_root)
    print("seeds.json 生成完成！")

# %%
import json
import shutil
from pathlib import Path

def convert_dataset(input_jsonl, output_root):
    """
    将原始JSONL数据集转换为instructpix2pix格式
    参数：
        input_jsonl: 输入JSONL文件路径
        output_root: 输出根目录路径
    """
    # 创建输出目录
    Path(output_root).mkdir(parents=True, exist_ok=True)
    
    # 读取JSONL文件
    with open(input_jsonl) as f:
        samples = [json.loads(line) for line in f]
    
    for idx, sample in enumerate(samples):
        # 创建样本目录
        sample_dir = Path(output_root) / f"sample_{idx}"
        sample_dir.mkdir(exist_ok=True)
        
        # 生成6位种子（前导零补齐）
        seed = f"{idx:06d}"
        
        try:
            # 复制并重命名输入输出图像
            shutil.copy(sample["input"], sample_dir / f"{seed}_0.jpg")
            shutil.copy(sample["output"], sample_dir / f"{seed}_1.jpg")
            
            # 写入prompt文件
            with open(sample_dir / "prompt.json", "w") as f:
                json.dump({"edit": sample["instruction"]}, f)
                
        except FileNotFoundError as e:
            print(f"警告：文件不存在，跳过样本 {idx}")
            shutil.rmtree(sample_dir)  # 删除空目录
            continue

if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    input_jsonl = "./instructpix2pix_dataset.jsonl"  # 您的原始JSONL文件路径
    output_root = "./"              # 输出目录
    
    convert_dataset(input_jsonl, output_root)
    print("数据集转换完成！")
# %%
