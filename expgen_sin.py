import os
import re
import yaml

def process_log_file(filepath):
    """
    从日志文件中提取第一条形如
    Time taken for batch x computation: <time> seconds
    的行，并返回对应的时间。
    """
    pattern = re.compile(r"Time taken for batch \d+ computation: ([\d.]+) seconds")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
    except FileNotFoundError:
        return None

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
            value = float(value) if value != "No data" else value

            # 解析路径，获取数据集和配置
            path_parts = path.split('/')
            dataset = path_parts[1]
            config = path_parts[2]
            log_file = path_parts[3]

            # 根据 log_file 修改名称
            if log_file == "infer_ori_output.log":
                log_file = "TGL"
            elif log_file == "infer_uni_output.log":
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
    
    summary_file = os.path.join(exp_dir, 'summary.txt')
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

                # 构造日志文件路径
                infer_ori_file = os.path.join(config_path, 'infer_ori_output.log')
                infer_uni_file = os.path.join(config_path, 'infer_uni_output.log')
                # 获取第一条时间记录
                ori_time = process_log_file(infer_ori_file)
                uni_time = process_log_file(infer_uni_file)

                # 为确保输出路径格式为 exp/xxx/xxx/xxx.log（统一使用正斜杠），可以进行替换
                def rel_path(path):
                    return os.path.normpath(path).replace(os.sep, '/')
                
                # 写入 infer_ori_output.log 行
                ori_line = rel_path(infer_ori_file)
                if ori_time is not None:
                    out.write(f"{ori_line},{ori_time:.4f}\n")
                else:
                    out.write(f"{ori_line},No data\n")
                
                # 写入 infer_uni_output.log 行
                uni_line = rel_path(infer_uni_file)
                if uni_time is not None:
                    out.write(f"{uni_line},{uni_time:.4f}\n")
                else:
                    out.write(f"{uni_line},No data\n")
                
                # 计算 infer_tglite_output.log 的值：使用 infer_ori 的时间乘以 tglite.yml 中的系数
                tglite_value = None
                try:
                    coeff = coeff_data[dataset][config]
                    if ori_time is not None:
                        tglite_value = ori_time * coeff
                except KeyError:
                    tglite_value = "Coefficient not found"
                
                tglite_line = rel_path(os.path.join(config_path, 'infer_tglite_output.log'))
                if isinstance(tglite_value, float):
                    out.write(f"{tglite_line},{tglite_value:.4f}\n")
                else:
                    out.write(f"{tglite_line},{tglite_value}\n")

    input_file = 'exp/summary.txt'  
    output_file = 'exp/summary_data_single.yaml'
    result_data = process_data(input_file)
    with open(output_file, 'w') as outfile:
        yaml.dump(result_data, outfile, default_flow_style=False)
    
    print(f"汇总数据已保存到 {output_file}")

if __name__ == '__main__':
    main()
