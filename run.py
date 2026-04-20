import os
import subprocess

# 定义数据集和配置文件列表
data_list = ['WIKITALK', 'GDELT', 'REDDIT', 'ROADCA', 'ROADTX']
config_list = ['config/APAN.yml', 'config/DySAT.yml', 'config/JODIE.yml', 'config/TGAT.yml', 'config/TGN.yml']

# 创建输出文件夹
exp_dir = 'exp'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# 遍历所有的数据集和配置文件
for data in data_list:
    for config in config_list:
        # 为每个组合创建对应的文件夹
        data_dir = os.path.join(exp_dir, data)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        config_name = config.split('/')[-1].split('.')[0]  # 获取config文件的名称（去掉路径和扩展名）
        config_dir = os.path.join(data_dir, config_name)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        # 记录开始运行 infer_ori.py
        start_message = f"Starting infer_ori_32GB.py with --data {data} --config {config}..."
        print(start_message)
        with open(os.path.join(config_dir, 'infer_ori_32GB_output.log'), 'w') as f:
            f.write(f"START: {start_message}\n")  # 记录开始信息
            subprocess.run(['python', 'infer_ori_32GB.py', '--data', data, '--config', config], stdout=f, stderr=subprocess.STDOUT)
        # 记录结束运行 infer_ori.py
        end_message = f"Finished infer_ori_32GB.py with --data {data} --config {config}."
        print(end_message)
        with open(os.path.join(config_dir, 'infer_ori_32GB_output.log'), 'a') as f:
            f.write(f"END: {end_message}\n")  # 记录结束信息

        # 记录开始运行 infer_uni.py
        start_message = f"Starting infer_uni_32GB.py with --data {data} --config {config}..."
        print(start_message)
        with open(os.path.join(config_dir, 'infer_uni_32GB_output.log'), 'w') as f:
            f.write(f"START: {start_message}\n")  # 记录开始信息
            subprocess.run(['python', 'infer_uni_32GB.py', '--data', data, '--config', config], stdout=f, stderr=subprocess.STDOUT)
        # 记录结束运行 infer_uni.py
        end_message = f"Finished infer_uni_32GB.py with --data {data} --config {config}."
        print(end_message)
        with open(os.path.join(config_dir, 'infer_uni_32GB_output.log'), 'a') as f:
            f.write(f"END: {end_message}\n")  # 记录结束信息

print("All tasks completed.")
