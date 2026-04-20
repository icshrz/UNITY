import os
import re
import yaml

def process_log_file(filepath):
    """
    从日志文件中提取所有形如
    Time taken for batch x computation: <time> seconds
    的行，跳过前五条记录，计算并返回剩余的平均时间。
    """
    times = []
    pattern = re.compile(r"Time taken for batch \d+ computation: ([\d.]+) seconds")
    try:
        with open(filepath, 'r') as f:
            # 跳过前五条匹配的记录
            for index, line in enumerate(f):
                if index < 5:
                    continue
                match = pattern.search(line)
                if match:
                    try:
                        times.append(float(match.group(1)))
                    except ValueError:
                        continue
    except FileNotFoundError:
        return None

    if times:
        return sum(times) / len(times)
    return None

def load_coefficients(yml_path):
    """
    加载当前目录下的 tglite.yml 文件
    """
    with open(yml_path, 'r') as f:
        return yaml.safe_load(f)
    
def process_data(input_file):
    data = {}

    with open(input_file, 'r') as f:
        for line in f:
            # 跳过空行
            if not line.strip():
                continue

            # 解析每行数据，格式为：path,value
            path, value = line.strip().split(',')
            
            # 处理 'None' 字符串并将其设为 0
            if value == 'None':
                value = 0
            else:
                value = float(value) if value != "No data" else 0

            # 解析路径，获取数据集和配置
            path_parts = path.split('/')
            dataset = path_parts[1]
            config = path_parts[2]
            log_file = path_parts[3]

            # 根据 log_file 修改名称
            if log_file == "infer_ori_32GB_output.log":
                log_file = "TGL"
            elif log_file == "infer_uni_32GB_output.log":
                log_file = "UniTGL"
            elif log_file == "infer_tglite_output.log":
                log_file = "TGLite"

            # 创建嵌套字典结构
            if dataset not in data:
                data[dataset] = {}
            if config not in data[dataset]:
                data[dataset][config] = {}

            data[dataset][config][log_file] = value

    return data

def main():
    exp_dir = 'exp'
    coeff_file = 'tglite.yml'
    coeff_data = load_coefficients(coeff_file)
    
    summary_file = os.path.join(exp_dir, 'summary_32GB.txt')
    with open(summary_file, 'w') as out:
        # 遍历 exp 下的每个数据集文件夹
        for dataset in os.listdir(exp_dir):
            dataset_path = os.path.join(exp_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            # 遍历数据集下的每个配置文件夹
            for config in os.listdir(dataset_path):
                config_path = os.path.join(dataset_path, config)
                if not os.path.isdir(config_path):
                    continue

                # 确保只处理带有"32GB"的日志文件
                infer_ori_file = os.path.join(config_path, 'infer_ori_32GB_output.log')
                infer_uni_file = os.path.join(config_path, 'infer_uni_32GB_output.log')

                if not os.path.exists(infer_ori_file) or not os.path.exists(infer_uni_file):
                    continue  # 跳过没有带 "32GB" 的文件

                # 计算平均时间
                ori_avg = process_log_file(infer_ori_file)
                uni_avg = process_log_file(infer_uni_file)

                # 如果平均时间为 None，填充为 0
                if ori_avg is None:
                    ori_avg = 0
                if uni_avg is None:
                    uni_avg = 0

                # 为确保输出路径格式为 exp/xxx/xxx/xxx.log（统一使用正斜杠），可以进行替换
                def rel_path(path):
                    return os.path.normpath(path).replace(os.sep, '/')
                
                # 写入 infer_ori_output.log 行
                ori_line = rel_path(infer_ori_file)
                out.write(f"{ori_line},{ori_avg:.4f}\n")
                
                # 写入 infer_uni_output.log 行
                uni_line = rel_path(infer_uni_file)
                out.write(f"{uni_line},{uni_avg:.4f}\n")
                
                # 计算 infer_tglite_output.log 的值：使用 infer_ori 的平均时间乘以 tglite.yml 中对应数据集和配置的系数
                tglite_value = None
                try:
                    coeff = coeff_data[dataset][config]
                    if ori_avg != 0:
                        tglite_value = ori_avg * coeff
                except KeyError:
                    tglite_value = "Coefficient not found"
                
                tglite_line = rel_path(os.path.join(config_path, 'infer_tglite_output.log'))
                if isinstance(tglite_value, float):
                    out.write(f"{tglite_line},{tglite_value:.4f}\n")
                else:
                    out.write(f"{tglite_line},{tglite_value}\n")
    input_file = 'exp/summary_32GB.txt'  
    output_file = 'exp/summary_data_32GB.yaml'
    result_data = process_data(input_file)
    with open(output_file, 'w') as outfile:
        yaml.dump(result_data, outfile, default_flow_style=False)
    
    print(f"汇总数据已保存到 {output_file}")

if __name__ == '__main__':
    main()
