import torch
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class VibrationDataset(Dataset):
    def __init__(self, params, signals):
        """
        初始化数据集
        :param params: 工艺参数 (num_samples, num_features)
        :param signals: 震动信号 (num_samples, seq_len)
        """
        self.params = torch.tensor(params, dtype=torch.float32)
        self.signals = torch.tensor(signals, dtype=torch.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: 数据索引
        :return: (params, signals)
        """
        return self.params[idx], self.signals[idx]

