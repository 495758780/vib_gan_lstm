import pandas as pd
import matplotlib.pyplot as plt


def plot_img():
    df = pd.read_excel('meg.xlsx')
    print(df)

    x = df.iloc[:, [0]]
    y = df.iloc[:, [1]]

    plt.plot(x, y)
    plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取数据
# df = pd.read_excel('meg.xlsx')
#
# # 选择数据段
# x = df.iloc[:, [0]]
# y = df.iloc[:, [1]]
#
# # 将数据转换为一维数组
# x = x.values.flatten()
# y = y.values.flatten()
#
# # 每100个点取一个点
# x_resampled = x[::100]  # 从原始数据中每隔100个点取一个
# y_resampled = y[::100]
#
# # 绘制降采样后的图形
# plt.figure(figsize=(10, 6))
# plt.plot(x_resampled, y_resampled, label='Downsampled Data', color='green', marker='o')
# plt.scatter(x, y, label='Original Data', color='blue', alpha=0.5, s=10)
# plt.title('Downsampled Data (Every 100th Point)')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(True)
# plt.show()






# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取 Excel 数据
# df = pd.read_excel('meg.xlsx')
#
# # 创建一个包含 8 个子图的绘图区域，2 行 4 列
# fig, axes = plt.subplots(2, 4, figsize=(15, 8))
#
# # 遍历列索引（从第 1 列到第 8 列）
# for i in range(8):
#     ax = axes[i // 4, i % 4]  # 确定当前子图的行和列索引
#     x = df.iloc[:, 0]         # X 轴为第 0 列
#     y = df.iloc[:, i + 1]     # Y 轴从第 1 列到第 8 列
#     ax.plot(x, y)             # 绘制当前子图
#     ax.set_title(f'Column {i + 1}')  # 设置子图标题
#
# # 调整布局
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
#
# # 读取 Excel 数据
# df = pd.read_excel('meg.xlsx')
#
# # 提取第一列数据
# x = df.iloc[20000:30000, 0]  # X 轴数据
# y = df.iloc[20000:30000, 1]  # Y 轴数据（第一列）
#
# # 对 Y 数据进行平滑处理
# y_smooth = savgol_filter(y, window_length=11, polyorder=3)  # 窗口大小为 11，3 次多项式拟合
#
# # 绘制原始曲线
# plt.plot(x, y, label='Original Curve', color='blue', linestyle='--', alpha=0.7)
#
# # 绘制平滑后的曲线
# plt.plot(x, y_smooth, label='Smoothed Curve', color='red')
#
# # 添加标题、图例和坐标轴标签
# plt.title('Original vs Smoothed Curve')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取 Excel 数据
# df = pd.read_excel('meg.xlsx')
#
# # 提取 X 和 Y 数据
# x = df.iloc[:, 0]  # X 轴数据
# y = df.iloc[:, 1]  # Y 轴数据（第一列）
#
# # 对 Y 数据每 100 个点求均值
# y_avg = y.rolling(window=1000).mean()  # 滑动窗口为 100，计算均值
#
# # 绘制原始曲线
# plt.plot(x, y, label='Original Curve', color='blue', linestyle='--', alpha=0.7)
#
# # 绘制均值平滑后的曲线
# plt.plot(x, y_avg, label='Averaged Curve (100 data points)', color='green')
#
# # 添加标题、图例和坐标轴标签
# plt.title('Original vs Averaged Curve (100 Data Points)')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

# 滑动平滑
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取 Excel 数据
# df = pd.read_excel('meg.xlsx')
#
# # 提取 X 和 Y 数据
# x = df.iloc[:, 0]  # X 轴数据
# y = df.iloc[:, 1]  # Y 轴数据（第一列）
#
# # 分割数据，分别处理大于 0 和小于 0 的部分
# y_positive = y[y > 0]
# x_positive = x[y > 0]
#
# y_negative = y[y < 0]
# x_negative = x[y < 0]
#
# # 对大于 0 的数据进行均值平滑
# y_positive_avg = y_positive.rolling(window=100).mean()
#
# # 对小于 0 的数据进行均值平滑
# y_negative_avg = y_negative.rolling(window=100).mean()
#
# # 创建图形
# plt.figure(figsize=(10, 6))
#
# # 绘制原始数据为散点图
# plt.scatter(x, y, label='Original Data', color='blue', alpha=0.5, s=10)
#
# # 绘制大于 0 的数据的均值平滑曲线
# plt.plot(x_positive, y_positive_avg, label='Smoothed Positive Data', color='green')
#
# # 绘制小于 0 的数据的均值平滑曲线
# plt.plot(x_negative, y_negative_avg, label='Smoothed Negative Data', color='red')
#
# # 添加标题、图例和坐标轴标签
# plt.title('Original Data (Scatter) and Smoothed Curves for Positive and Negative Data')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取 Excel 数据
# df = pd.read_excel('meg.xlsx')
#
# # 提取 X 和 Y 数据
# x = df.iloc[:, 0]  # X 轴数据
# y = df.iloc[:, 1]  # Y 轴数据（第一列）
#
# # 将数据分为正数和负数两部分
# y_positive = y[y > 0]
# x_positive = x[y > 0]
#
# y_negative = y[y < 0]
# x_negative = x[y < 0]
#
# # 每 100 个点取一个均值（正数和负数分别处理）
# window_size = 100
# y_positive_avg = y_positive.groupby(y_positive.index // window_size).mean()  # 正数部分均值
# x_positive_avg = x_positive.iloc[::window_size].reset_index(drop=True)  # 正数部分的 X 轴采样
#
# y_negative_avg = y_negative.groupby(y_negative.index // window_size).mean()  # 负数部分均值
# x_negative_avg = x_negative.iloc[::window_size].reset_index(drop=True)  # 负数部分的 X 轴采样
#
# # 计算总采样点数为原数据点数的 1%
# sampling_rate = 0.01
# total_points_positive = int(len(x_positive) * sampling_rate)
# total_points_negative = int(len(x_negative) * sampling_rate)
#
# # 计算每部分的采样间隔
# step_positive = len(x_positive_avg) // total_points_positive
# step_negative = len(x_negative_avg) // total_points_negative
#
# # 根据采样间隔选择采样点
# x_positive_resampled = x_positive_avg[::step_positive]
# y_positive_resampled = y_positive_avg[::step_positive]
#
# x_negative_resampled = x_negative_avg[::step_negative]
# y_negative_resampled = y_negative_avg[::step_negative]
#
# # 确保最终的采样点数与计算的匹配
# x_positive_resampled = x_positive_resampled[:total_points_positive]
# y_positive_resampled = y_positive_resampled[:total_points_positive]
#
# x_negative_resampled = x_negative_resampled[:total_points_negative]
# y_negative_resampled = y_negative_resampled[:total_points_negative]
#
# # 创建图形
# plt.figure(figsize=(10, 6))
#
# # 绘制原始数据为散点图
# plt.scatter(x, y, label='Original Data', color='blue', alpha=0.5, s=10)
#
# # 绘制正数部分采样后的均值曲线
# plt.plot(x_positive_resampled, y_positive_resampled, label='Resampled Positive Data', color='green', marker='o')
#
# # 绘制负数部分采样后的均值曲线
# plt.plot(x_negative_resampled, y_negative_resampled, label='Resampled Negative Data', color='red', marker='o')
#
# # 添加标题、图例和坐标轴标签
# plt.title('Original Data and Resampled Positive & Negative Curves (Every 100 Points)')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()





