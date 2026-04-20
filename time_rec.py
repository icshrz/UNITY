import time
import subprocess
from datetime import datetime

# 输出文件路径
output_file = "memory_usage_log.txt"

def get_memory_usage():
    # 使用 free 命令来获取内存和 swap 信息
    result = subprocess.run(['free', '-m'], stdout=subprocess.PIPE)
    result_str = result.stdout.decode('utf-8')

    # 解析输出并提取内存和 swap 使用情况
    lines = result_str.splitlines()
    memory_line = lines[1].split()  # 第2行是内存使用情况
    swap_line = lines[2].split()    # 第3行是 swap 使用情况
    
    total_memory = memory_line[1]
    used_memory = memory_line[2]
    free_memory = memory_line[3]
    
    total_swap = swap_line[1]
    used_swap = swap_line[2]
    free_swap = swap_line[3]
    
    return total_memory, used_memory, free_memory, total_swap, used_swap, free_swap

def record_memory_usage():
    # 打开文件以写入内存使用情况
    with open(output_file, 'a') as f:
        while True:
            # 获取当前时间和内存、swap 使用情况
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            total_mem, used_mem, free_mem, total_swap, used_swap, free_swap = get_memory_usage()
            
            # 格式化字符串并写入文件
            log_entry = (f"{current_time} | "
                         f"Total Memory: {total_mem}MB | Used Memory: {used_mem}MB | Free Memory: {free_mem}MB | "
                         f"Total Swap: {total_swap}MB | Used Swap: {used_swap}MB | Free Swap: {free_swap}MB\n")
            f.write(log_entry)
            
            # 等待0.5秒再获取下一次内存数据
            time.sleep(0.5)

if __name__ == "__main__":
    try:
        print(f"Starting memory and swap usage logging. Data will be saved to '{output_file}'...")
        record_memory_usage()
    except KeyboardInterrupt:
        print("Memory and swap usage logging stopped.")
