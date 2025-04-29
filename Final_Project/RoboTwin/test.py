import os
import json
import random
from tqdm import tqdm

# 定义任务与指令的映射
TASK_TO_INSTRUCTION = {
    "block-hammer-beat": "beat the block with the hammer",
    "block-handover": "handover the blocks",
    "blocks-stack-easy": "stack blocks"
}

def generate_dataset_jsonl(root_dir, output_jsonl="dataset.jsonl"):
    data_entries = []
    
    # 遍历每个任务文件夹
    for task in os.listdir(root_dir):
        task_dir = os.path.join(root_dir, task)
        if not os.path.isdir(task_dir) or task not in TASK_TO_INSTRUCTION:
            continue
        
        instruction = TASK_TO_INSTRUCTION[task]
        
        # 遍历每个episode
        for episode in tqdm(os.listdir(task_dir), desc=f"Processing {task}"):
            episode_dir = os.path.join(task_dir, episode)
            if not os.path.isdir(episode_dir):
                continue
            
            # 获取episode中的所有图片文件
            image_files = [f for f in os.listdir(episode_dir) if f.endswith(".jpg")]
            
            # 提取帧数
            frame_numbers = [int(f.split(".")[0]) for f in image_files]
            
            if not frame_numbers:
                continue  # 如果没有图片，跳过
            
            # 随机选择输入帧
            input_frame = random.choice(range(min(frame_numbers), min(50, max(frame_numbers))))
            input_path = os.path.join(episode_dir, f"{input_frame}.jpg")
            
            # 随机选择输出帧
            last_frames = sorted(frame_numbers)[-50:]  # 获取最后50帧
            target_frame = random.choice(last_frames)
            target_path = os.path.join(episode_dir, f"{target_frame}.jpg")
            
            # 添加数据条目
            data_entries.append({
                "input": input_path,
                "output": target_path,
                "instruction": instruction
            })
    
    # 保存为JSONL文件
    with open(output_jsonl, "w") as f:
        for entry in data_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"生成完成！共 {len(data_entries)} 条数据，保存至 {output_jsonl}")

# 调用函数（修改为你的实际路径）
generate_dataset_jsonl(
    root_dir="/home/zcai/Downloads/Final_Project/RoboTwin",  # 替换为你的RoboTwin路径
    output_jsonl="instructpix2pix_dataset.jsonl"
)

# 生成 300 个随机种子（对应 300 个样本）
# seeds = [random.randint(0, 2**32 - 1) for _ in range(300)]
# with open("data/instructpix2pix/seeds.json", "w") as f:
#     json.dump(seeds, f)