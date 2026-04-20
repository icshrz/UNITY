import ctypes
import numpy as np
import cupy as cp
from ctypes import POINTER, c_float

class Mapping:
    def __init__(self, ptrs):
        self.ptrs = ptrs

    def combine_to_tensor(self):
        # 组合多个指针数据为一个连续的张量
        total_size = sum(len(ptr) for ptr in self.ptrs)
        combined_data = np.empty(total_size, dtype=np.float32)

        idx = 0
        for ptr in self.ptrs:
            block_size = len(ptr)
            combined_data[idx: idx + block_size] = np.frombuffer(ctypes.cast(ptr, POINTER(c_float)), dtype=np.float32, count=block_size)
            idx += block_size

        return combined_data

    def to_gpu(self):
        # 转换为 CUDA 统一内存
        combined_data = self.combine_to_tensor()
        gpu_array = cp.asarray(combined_data)
        return gpu_array
