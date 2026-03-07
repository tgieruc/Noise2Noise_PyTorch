import torch
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.datasets = torch.cat([dataset1[:, None], dataset2[:, None]], dim=1)
        self.transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        )

    def __getitem__(self, i):
        if torch.rand(1) > 0.5:
            return self.transforms(self.datasets[i])
        else:
            return self.transforms(self.datasets[i, [1, 0]])

    def __len__(self):
        return len(self.datasets)
