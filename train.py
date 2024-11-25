import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import model
import pandas as pd
import os
import my_dataset

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
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# 创建生成器和判别器
generator = model.Generator(input_dim=21, output_dim=50)
discriminator = model.Discriminator(input_dim=21, sequence_length=50)

# 损失函数
criterion = nn.BCELoss()

# 优化器
lr = 0.0002
generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


def train_step(real_data, condition, generator, discriminator, criterion):
    batch_size = real_data.size(0)

    # 生成器训练
    generated_data = generator(real_data)  # 生成的震动时序数据

    # 判别器的真实数据输出
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    discriminator_optimizer.zero_grad()

    # 判别器计算损失
    real_output = discriminator(torch.cat((real_data, condition), 1))
    real_loss = criterion(real_output, real_labels)

    fake_output = discriminator(torch.cat((generated_data.detach(), condition), 1))
    fake_loss = criterion(fake_output, fake_labels)

    disc_loss = (real_loss + fake_loss) / 2
    disc_loss.backward()
    discriminator_optimizer.step()

    # 生成器训练
    generator_optimizer.zero_grad()

    fake_output = discriminator(torch.cat((generated_data, condition), 1))
    gen_loss = criterion(fake_output, real_labels)  # 生成器希望判别器认为它生成的是真实的

    gen_loss.backward()
    generator_optimizer.step()

    return gen_loss.item(), disc_loss.item()


epochs = 20
def train():
    for epoch in range(epochs):
        for i, (real_data, condition) in enumerate(dataloader):

            gen_loss, disc_loss = train_step(real_data, condition, generator, discriminator, criterion)

        print(f'Epoch [{epoch + 1}/{epochs}], Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}')

if __name__ == '__main__':
    train()
