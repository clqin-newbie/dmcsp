import torch
from torch.utils.data import Dataset
from pymatgen.core.composition import Composition
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import numpy as np 
class CspDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        # raw_data = raw_data.iloc[:int(raw_data.shape[0]/20), :]
        self.fps = torch.tensor(data.iloc[:, :-4].values, dtype=torch.float32)
        self.r0 = torch.tensor(data.iloc[:, -4:-3].values, dtype=torch.float32)
        self.r0s = torch.tensor(data.iloc[:, -3:-1].values, dtype=torch.float32)
        self.lables = self.get_lablels(data['Z'])

    def get_lablels(self, labels):
        unique_element = 94
        # 初始化全0矩阵
        one_hot = torch.zeros((len(labels), unique_element), dtype=torch.float32)

        # 核心：按索引把对应列置1
        one_hot[torch.arange(len(labels)), labels-1] = 1
        # hot_label = torch.tensor(encoded, dtype=torch.float32)
        # print(unique_elements)
        return one_hot
    
    def __len__(self) -> int:
        return self.fps.shape[0]

    def __getitem__(self, index: int):
        r = self.fps[index]
        r0 = self.r0[index]
        r0s = self.r0s[index]
        y = self.lables[index]
        return r, r0, r0s, y