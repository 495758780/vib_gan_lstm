import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_dim)
        self.batch_norm = nn.BatchNorm1d(256)

    def forward(self, x):
        # 确保输入的形状正确，例如 x.shape = (batch_size, input_dim)
        x = torch.relu(self.fc1(x))
        x = self.batch_norm(x)  # BatchNorm 对第 2 维进行标准化
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # 生成数据范围 (-1, 1)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim + sequence_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # 输出一个0到1之间的值，表示真假
        return x
