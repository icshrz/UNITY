import yaml

# 读取原始 YAML 文件
input_file = "gpu_uni.yaml"
output_file = "gpu_uni2.yaml"

# 从 YAML 文件读取数据
with open(input_file, "r") as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# 修改数据：将时间戳从 0 开始，并将其除以 10
base_time = float(data[0]['timestamp'])  # 获取原始时间戳的起始值
for i, item in enumerate(data):
    # 将时间戳归零并除以 10
    item['timestamp'] = str((float(item['timestamp']) - base_time) / 10)

# 将修改后的数据写入新的 YAML 文件
with open(output_file, "w") as file:
    yaml.dump(data, file, default_flow_style=False)

print(f"Modified data has been saved to {output_file}.")
