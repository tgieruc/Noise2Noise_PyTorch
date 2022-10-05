import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class ConvBlockLeakyRelu(nn.Module):
    '''
    A block containing a Conv2d followed by a leakyRelu
    '''

    def __init__(self, chanel_in, chanel_out, kernel_size, stride=1, padding=1):
        super(ConvBlockLeakyRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(chanel_in, chanel_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ## ------- ENCODER -------
        self.enc_conv01 = nn.Sequential(
            ConvBlockLeakyRelu(3, 48, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv2 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv3 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv4 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv56 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
        )

        ## ------- DECODER -------
        self.dec_conv5ab = nn.Sequential(
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv4ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv3ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv2ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv1abc = nn.Sequential(
            ConvBlockLeakyRelu(99, 64, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(64, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, 3, 3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        ## ------- ENCODER -------
        residual_connection = [x]

        x = self.enc_conv01(x)
        residual_connection.append(x)

        x = self.enc_conv2(x)
        residual_connection.append(x)

        x = self.enc_conv3(x)
        residual_connection.append(x)

        x = self.enc_conv4(x)
        residual_connection.append(x)

        x = self.enc_conv56(x)

        ## ------- DECODER -------
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv5ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv4ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv3ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv2ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv1abc(x)

        return x


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


class Model():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = Net()
        self.net.to(self.device)

        self.optim = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.8)
        self.loss = nn.MSELoss()

    def load_pretrained_model(self, path) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, train_input, train_target, num_epochs, batch_size=8, num_workers=2) -> None:
        '''
        Trains the model, shows the progression with TQDM progress bar
        :param train_input: the input images in the form of a tensor
        :param train_target: the target images in the form of a tensor
        :param num_epochs: the number of epochs of training
        :param batch_size: the batch size
        :param num_workers: the number of workers for dataloading
        '''

        train_loader = torch.utils.data.DataLoader(
            Dataset(
                train_input,
                train_target
            ),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        self.net.train()

        for epoch in range(0, num_epochs):
            loop = tqdm(train_loader)
            train_loss = []
            for i, data in enumerate(loop):
                source, target = data[:, 0].float().cuda() / 255, data[:, 1].float().cuda() / 255
                denoised = self.net(source)

                loss_ = self.loss(denoised, target)
                train_loss.append(loss_.detach().cpu().item())

                self.optim.zero_grad()
                loss_.backward()
                self.optim.step()
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=np.mean(train_loss))

    def predict(self, test_input) -> torch.Tensor:
        '''
        Denoise the input tensor
        :param test_input: tensor of size (N1 , C, H, W), of images whose values are between 0 and 255, that has to be denoised by the network
        :return: the desnoised images, already normalized between 0 and 255
        '''
        self.net.eval()

        def normalization_cut(imgs):
            imgs_shape = imgs.shape
            imgs = imgs.flatten()
            imgs[imgs < 0] = 0
            imgs[imgs > 1] = 1
            imgs = imgs.reshape(imgs_shape)
            return imgs

        return 255 * normalization_cut(self.net((test_input / 255).to(self.device)))

    def save(self, path):
        torch.save(self.net.state_dict(), path)
