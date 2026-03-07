import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from noise2noise.dataset import Dataset
from noise2noise.net import Net


class Model:
    def __init__(self, lr=0.001) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = Net()
        self.net.to(self.device)

        self.optim = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.8)
        self.loss = nn.MSELoss()

    def load_pretrained_model(self, path) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, train_input, train_target, num_epochs, batch_size=8, num_workers=2) -> None:
        """Train the model, showing progression with a TQDM progress bar.

        Args:
            train_input: Input noisy images as a tensor.
            train_target: Target noisy images as a tensor.
            num_epochs: Number of training epochs.
            batch_size: Batch size for data loading.
            num_workers: Number of workers for data loading.
        """
        train_loader = DataLoader(
            Dataset(train_input, train_target),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        scheduler = CosineAnnealingLR(self.optim, T_max=num_epochs)
        self.net.train()

        for epoch in range(0, num_epochs):
            loop = tqdm(train_loader)
            train_loss = []
            for i, data in enumerate(loop):
                source = data[:, 0].float().to(self.device) / 255
                target = data[:, 1].float().to(self.device) / 255
                denoised = self.net(source)

                loss_ = self.loss(denoised, target)
                train_loss.append(loss_.detach().cpu().item())

                self.optim.zero_grad()
                loss_.backward()
                self.optim.step()
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=np.mean(train_loss))
            scheduler.step()

    def predict(self, test_input) -> torch.Tensor:
        """Denoise the input tensor.

        Args:
            test_input: Tensor of size (N, C, H, W) with values in [0, 255].

        Returns:
            Denoised images, normalized between 0 and 255.
        """
        self.net.eval()
        with torch.no_grad():
            output = self.net((test_input / 255).to(self.device))
        return 255 * torch.clamp(output, 0, 1)

    def save(self, path):
        torch.save(self.net.state_dict(), path)
