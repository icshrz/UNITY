import re
import yaml

# 读取原始数据文件
input_file = "gpu_usage2.txt"
output_file = "exp/gpu_uni.yaml"

# 正则表达式提取时间戳和GPU利用率（假设 GPU 利用率在 'GR3D_FREQ' 部分）
pattern = r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}) .+ GR3D_FREQ (\d+)%'

# 初始化时间戳和GPU利用率列表
data = []

# 读取文件并提取数据
with open(input_file, "r") as file:
    base_time = 0  # 从0秒开始
    for i, line in enumerate(file):
        match = re.search(pattern, line)
        if match:
            timestamp = match.group(1)
            gpu_usage = int(match.group(2))  # 获取GPU的使用百分比
            # 计算时间戳并存入列表
            time_in_seconds = base_time + (i * 0.1)  # 每次间隔0.1s
            data.append({
                'timestamp': f"{time_in_seconds:.1f}",
                'gpu_usage': gpu_usage
            })

# 将数据写入YAML文件
with open(output_file, "w") as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)

print(f"Data has been extracted and saved to {output_file}.")
