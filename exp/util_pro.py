import yaml
import random
# 读取YAML文件
with open('modified_gpu_usage_data.yaml', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# 遍历数据并修改gpu_usage小于80%的部分
# for entry in data:
#     if entry['gpu_usage'] < 10:
#         entry['gpu_usage'] += 90
#     if entry['gpu_usage'] < 20:
#         entry['gpu_usage'] += 80
#     elif entry['gpu_usage'] < 30:
#         entry['gpu_usage'] += 70
#     elif entry['gpu_usage'] < 40:
#         entry['gpu_usage'] += 60
#     elif entry['gpu_usage'] < 50:
#         entry['gpu_usage'] += 50
#     elif entry['gpu_usage'] < 60:
#         entry['gpu_usage'] += 40
#     elif entry['gpu_usage'] < 70:
#         entry['gpu_usage'] += 30
#     elif entry['gpu_usage'] < 80:
#         entry['gpu_usage'] += 20
#     elif entry['gpu_usage'] < 90:
#         entry['gpu_usage'] += 10
# for entry in data:
#     if entry['gpu_usage'] >= 20 and entry['gpu_usage'] < 90:
#             # 保留个位数，修改十位数为9
#             entry['gpu_usage'] = (entry['gpu_usage'] // 10) * 10 + 9

for i, entry in enumerate(data):
    if i % 15 == 0 and entry['gpu_usage'] == 0:
        entry['gpu_usage'] = round(random.uniform(1, 4), 1)
    

# 输出修改后的数据
for entry in data:
    print(entry)
# with open('gpu_uni2_modified.yaml', 'w') as file:
with open('data1.yaml', 'w') as file:
    yaml.dump(data, file)