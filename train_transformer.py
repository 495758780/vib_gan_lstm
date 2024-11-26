import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import model
import pandas as pd
import os
import my_dataset
import model_transformer
import matplotlib.pyplot as plt
from IPython.display import clear_output

dataset_gan_x = pd.read_csv('data/gan/process_parameters.csv', dtype=float)
dataset_gan_x = dataset_gan_x.values
dataset_gan_y = pd.DataFrame()
for a, b, c in os.walk('data/gan/output'):
    for index, c1 in enumerate(c):
        for i in range(8):
            temp_df = pd.read_csv(os.path.join(a, c1)).iloc[0:50, [2+i]]
            temp_df = temp_df.transpose()
            dataset_gan_y = pd.concat([dataset_gan_y, temp_df], axis=0)
dataset_gan_y = dataset_gan_y.values
dataset = my_dataset.VibrationDataset(dataset_gan_x, dataset_gan_y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'device is {device}')

# 参数设置
input_dim = 21         # 初始参数维度
embed_dim = 64         # 嵌入维度
num_heads = 4          # 注意力头数
ff_dim = 128           # 前馈网络隐藏层大小
num_decoder_layers = 3 # 解码器层数
output_seq_len = 50    # 输出序列长度

# 初始化模型
model = model_transformer.TransformerTimeSeries(
    input_dim=input_dim,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_decoder_layers=num_decoder_layers,
    output_seq_len=output_seq_len
)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 初始化绘图
plt.ion()  # 开启交互模式
fig, ax = plt.subplots(figsize=(10, 6))
losses = []
line, = ax.plot([], [], label="Loss")  # 初始化空曲线
ax.set_xlim(0, 50)  # 设置横轴范围（可以根据 epoch 动态调整）
ax.set_ylim(0, 1)   # 设置纵轴范围（可以根据实际损失动态调整）
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Real-time Loss Curve")
ax.legend()
ax.grid()


model.train()
model.to(device)

epochs = 50
for epoch in range(epochs):
    for sec, target in dataloader:
        optimizer.zero_grad()

        # 前向计算
        predictions = model(sec.to(device))
        loss = criterion(predictions, target.squeeze(-1).to(device))

        # 反向传播与参数更新
        loss.backward()
        optimizer.step()

        # 更新损失记录
        losses.append(loss.item())

        # 动态更新曲线数据
        line.set_data(range(len(losses)), losses)  # 更新曲线数据
        ax.set_xlim(0, len(losses))  # 动态调整横轴范围
        ax.set_ylim(0, max(losses) * 1.1)  # 动态调整纵轴范围（留出空间）

        # 刷新图像
        fig.canvas.draw()
        fig.canvas.flush_events()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

plt.ioff()  # 关闭交互模式
plt.show()
