import tracemalloc
import numpy as np
import time

def run_fft_and_measure_memory(data_size):
    """
    执行 FFT 运算并测量其内存使用。
    """
    print(f"--- 正在处理数据大小为 {data_size} 的 FFT ---")

    # 1. 启用 tracemalloc
    # 在实际应用中，你可能在脚本的入口处启用一次
    tracemalloc.start()

    # 2. 获取初始快照
    # 确保在执行FFT核心代码之前获取快照，排除之前的内存分配
    snapshot_before_fft = tracemalloc.take_snapshot()

    # 准备数据
    # 创建一个复数数组作为输入，模拟实际的FFT输入数据
    # 例如，一个长度为 data_size 的一维数组
    data = np.random.rand(data_size) + 1j * np.random.rand(data_size)

    # 记录 FFT 开始时间
    start_time = time.time()

    # 3. 执行 FFT 算法
    # 这里使用 NumPy 的 FFT 函数
    fft_result = np.fft.fft(data)

    # 记录 FFT 结束时间
    end_time = time.time()

    # 4. 获取最终快照
    snapshot_after_fft = tracemalloc.take_snapshot()

    # 5. 比较快照并分析
    # 计算两个快照之间的内存差异
    top_stats = snapshot_after_fft.compare_to(snapshot_before_fft, 'lineno')

    print(f"FFT 运算耗时: {(end_time - start_time):.4f} 秒")
    print("\n--- FFT 运算期间的内存分配统计 (前10条) ---")
    for stat in top_stats[:10]:
        # stat.traceback 包含了文件和行号信息
        # stat.size 内存大小 (bytes)
        # stat.count 分配次数
        print(f"{stat.traceback.format(limit=1)[0]}\n"
              f"  内存增长: {stat.size / 1024:.2f} KB, "
              f"分配次数: {stat.count}")

    # 可以进一步筛选出与 numpy.fft 相关的内存分配
    # 例如，只看包含 'numpy' 的文件
    numpy_fft_memory = 0
    for stat in top_stats:
        if "numpy" in str(stat.traceback):
            numpy_fft_memory += stat.size
    print(f"\n--- 估计 NumPy FFT 相关的总内存增长: {numpy_fft_memory / 1024:.2f} KB ---")

    # 6. 禁用 tracemalloc (如果不再需要)
    tracemalloc.stop()
    print("\n----------------------------------------\n")

# 运行测试，尝试不同大小的数据
run_fft_and_measure_memory(1024)      # 1K点FFT
run_fft_and_measure_memory(1024 * 8)  # 8K点FFT
run_fft_and_measure_memory(1024 * 64) # 64K点FFT