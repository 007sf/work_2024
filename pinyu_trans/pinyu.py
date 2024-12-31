import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks
import os


def detect_turning_points(data):
    """
    检测一维数据中的所有拐点，包括上升、下降、平坦区的起点或终点。

    参数:
    - data: 一维数组或列表，输入数据

    返回:
    - turning_points: 拐点的索引列表
    """
    data = np.array(data)  # 确保输入是 NumPy 数组
    diff = np.diff(data)  # 计算相邻点的差分
    turning_points = [0]  # 第一个点总是拐点

    # 检测拐点
    for i in range(1, len(diff)):
        if (diff[i] > 0 > diff[i - 1]) or (diff[i] < 0 < diff[i - 1]) or (diff[i] == 0 and diff[i - 1] != 0):
            turning_points.append(i)

    turning_points.append(len(data) - 1)  # 最后一个点也是拐点
    return turning_points


matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
grid_size = 7
sensor_number = 49
second_maxima = np.zeros(sensor_number)
SMS = []
# 初始化一部字典，用于存储每种传感器数量下的幅值曲线
sensor_curves = {i: [] for i in range(1, sensor_number + 1)}
pic_dir = "../pic"
# for ii in range(20):
#
#     for i in range(1, 40):
#         grid_data = np.zeros((grid_size, grid_size))
#         selected_positions = np.random.permutation(grid_size ** 2)[:i]
#         grid_data.flat[selected_positions] = 1
#
#         fft_data = np.fft.fft2(grid_data)
#         fft_shifted = np.fft.fftshift(fft_data)
#         magnitude = np.abs(fft_shifted)
#         denominator = magnitude.max() - magnitude.min()
#         if denominator == 0:
#             normalized_magnitude = np.zeros_like(magnitude)
#         else:
#             normalized_magnitude = (magnitude - magnitude.min()) / denominator
#         magnitude = normalized_magnitude
#         sorted_magnitude = np.sort(magnitude.flatten())[::-1]
#         second_maxima[i - 1] = sorted_magnitude[1]
#
#
#         sensor_curves[i].append(magnitude.flatten())
#     SMS.append(second_maxima.copy())

from work_rl.env.grid_world_in_sssf import GridWorld

env = GridWorld()
state = {
    "position": (2, 0),
    "marks_left": 8,
    # "marks_pos":[(2,0),(5,0),(0,1),(3,1),(4,1),(6,1),
    #              (0,2),(1,2),(3,2),(4,2),(5,2),
    #              (3,3),(4,3),(6,3),
    #              (0,4),(1,4),(2,4),(4,4),
    #              (2,5),(3,5),(4,5),(6,5),
    #              (0,6),(4,6)]
    "marks_pos": [(0,6)]
    # "marks_pos":[(i,j) for i in range(7) for j in range(7)]
}


def __reward_mark_function(state):
    sensor_pos = state["marks_pos"]
    grid_data = np.zeros((env.env_size[0], env.env_size[1]))
    for sensor in sensor_pos:
        sensor_index = sensor[0] + sensor[1] * env.env_size[0]
        grid_data.flat[sensor_index] = 1

    fft_data = np.fft.fft2(grid_data)
    fft_shifted = np.fft.fftshift(fft_data)
    magnitude = np.abs(fft_shifted)
    # magnitude = np.abs(fft_data)
    # 归一化
    denominator = magnitude.max() - magnitude.min()
    if denominator == 0:
        normalized_magnitude = np.zeros_like(magnitude)
    else:
        normalized_magnitude = (magnitude - magnitude.min()) / denominator
    magnitude = normalized_magnitude
    print("这是归一化后的幅值：{}".format(magnitude))

    magnitude = magnitude.flatten()
    peaks = detect_turning_points(magnitude)
    peak_values = np.array(magnitude[peaks])
    # peak_values = np.unique(peak_values)[::-1]

    print(peak_values)

    ratios = []
    peak_values_sort = np.sort(peak_values)[::-1]
    for i in range(1, len(peak_values_sort)):
        ratio = (peak_values_sort[i - 1] - peak_values_sort[i]) / peak_values_sort[i - 1]
        ratios.append(ratio)

    # 4. 输出结果
    print("波峰位置:", peaks)
    print("波峰高度:", peak_values_sort)
    print("相邻波峰之间的差距比例:", ratios)

    # 5. 可视化
    plt.figure()
    hot_magnitude = magnitude.reshape((7, 7))
    plt.imshow(hot_magnitude, cmap='gray', )
    plt.colorbar(label='Magnitude')

    plt.figure()
    plt.plot(magnitude, label="Data")
    plt.plot(magnitude[peaks], label="Peaks")
    plt.legend()
    plt.title("Wave Peaks and Differences")
    plt.show()

    # return 2*(r_loc/sum)+0.05*(r_num/sum)+0.05*(r_time/sum)
    # 画幅值曲线
    plt.figure(figsize=(10, 6))
    plt.plot(magnitude.flatten(), alpha=0.7)
    plt.show()

#
#
a = __reward_mark_function(state)
print(a)

# 绘制相同传感器数量下的幅值曲线
# for i in range(1, sensor_number + 1):
#     if sensor_curves[i]:  # 检查是否有曲线数据
#         plt.figure(figsize=(10, 6))
#         for idx, curve in enumerate(sensor_curves[i]):
#             plt.plot(curve, alpha=0.7, label=f'实验 {idx + 1}')
#         plt.title(f'传感器数量为 {i} 时的傅里叶幅值曲线')
#         plt.xlabel('频率索引')
#         plt.ylabel('幅值')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(pic_dir, f"{i}_sensors_fourier_magnitude_curves.png"))
#         plt.close()
#
#
# # 定义x和y轴的刻度
# x_ticks = np.arange(1, grid_size + 1)
# y_ticks = np.arange(1, grid_size + 1)
#
#
#
# # 绘制归一化幅值的热图
# plt.figure()
# plt.imshow(normalized_magnitude, cmap='gray', extent=(0.5, 7.5, 0.5, 7.5))
# plt.colorbar()
# plt.axis('equal')
# plt.xlim(0.5, 7.5)
# plt.ylim(0.5, 7.5)
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
# plt.legend([])
# plt.savefig(os.path.join(pic_dir,f"fourier_spectrum_{sensor_number}_sensor.png"))
#
# # 绘制幅值数据的曲线图
# # plt.figure()
# # plt.plot(magnitude.flatten())
# # plt.legend([])
# # plt.savefig(os.path.join(pic_dir,f"fourier_magnitude_{sensor_number}_sensor.png"))
#
# # 绘制原始网格数据的热图
# plt.figure()
# plt.imshow(grid_data, cmap='gray', extent=(0.5, 7.5, 0.5, 7.5))
# plt.axis('equal')
# plt.xlim(0.5, 7.5)
# plt.ylim(0.5, 7.5)
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
# plt.legend([])
# plt.savefig(os.path.join(pic_dir,f"grid_7_{sensor_number}_sensor.png"))
#
# # 绘制第二大幅值的曲线图
# plt.figure()
# plt.plot(second_maxima)
# plt.legend([])
# plt.savefig(os.path.join(pic_dir,f"second_maxima_{sensor_number}_sensor.png"))
#
# plt.figure()
# plt.imshow(SMS, aspect='auto', cmap='viridis')
# plt.colorbar(label='Second Maxima')
# plt.xlabel('Sensor Number')
# plt.ylabel('Experiment')
# plt.xticks(ticks=np.arange(0, sensor_number, 5), labels=np.arange(1, sensor_number + 1, 5))
# plt.yticks(ticks=np.arange(10), labels=np.arange(1, 11))
# plt.title('Second Maxima across Experiments and Sensor Numbers')
# plt.savefig(os.path.join(pic_dir, "SMS_heatmap.png"))
