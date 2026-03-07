import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockLeakyRelu(nn.Module):
    """A block containing a Conv2d followed by a LeakyReLU."""

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=1):
        super(ConvBlockLeakyRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size, stride=stride, padding=padding, bias=False),
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
            nn.MaxPool2d(2),
        )

        self.enc_conv2 = nn.Sequential(ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1), nn.MaxPool2d(2))

        self.enc_conv3 = nn.Sequential(ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1), nn.MaxPool2d(2))

        self.enc_conv4 = nn.Sequential(ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1), nn.MaxPool2d(2))

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
            nn.Conv2d(32, 3, 3, stride=1, padding=1, bias=True),
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
