import ctypes
import numpy as np

# 模拟多个不连续的数据块
data1 = (ctypes.c_float * 128)(*range(128))
data2 = (ctypes.c_float * 256)(*range(128, 384))
data3 = (ctypes.c_float * 128)(*range(384, 512))

# 获取每个数据块的指针
ptr1 = ctypes.pointer(data1)
ptr2 = ctypes.pointer(data2)
ptr3 = ctypes.pointer(data3)
print(ptr1)
print(ptr2)
print(ptr3)

# 假设总的内存大小
total_size = len(data1) + len(data2) + len(data3)

# 通过ctypes.cast将每个数据块的指针拼接起来
# 注意：这里只是为了示例，并非实际内存操作，Python需要访问的数据块视为连续
combined_ptr = ptr1  # 先从第一个数据块开始
combined_array = np.ctypeslib.as_array(ctypes.cast(combined_ptr, ctypes.POINTER(ctypes.c_float)), shape=(total_size,))

# 在numpy中访问这些数据
print(combined_array)
