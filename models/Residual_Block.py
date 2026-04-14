import torch 
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Reusable defintion of the Residual model"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride= 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Computes the forward pass for the Resnet model"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)

        return out
    
if __name__ == "__main__":
    import os

    block = ResidualBlock(in_channels=64, out_channels=64, stride=1)
    x = torch.randn(1, 64, 32, 32)
    y = block(x)

    os.system('clear')
    print(f'Input shape: {x.shape}')
    print(f'Out shape: {y.shape}')

    block_2 = ResidualBlock(in_channels=64, out_channels=28, stride=1)
    y_2 = block_2(x)
    print(f'Out shape: {y_2.shape}')



    