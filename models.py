import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class InceptionModule(nn.Module):
    """Simplified Inception module with multiple filter sizes"""
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

        # 5x5 convolution branch (implemented as two 3x3 convs)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)

class MS_RAN(nn.Module):
    """Multi-Scale Residual Attention Network architecture combining residual connections, multi-scale feature extraction, and attention mechanisms"""
    def __init__(self, num_classes=10):
        super(MS_RAN, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual blocks
        self.res_block1 = ResidualBlock(32, 64, stride=2)  # Output: 64 x 14 x 14
        self.res_block2 = ResidualBlock(64, 128, stride=2)  # Output: 128 x 7 x 7

        # Inception module
        self.inception = InceptionModule(128, 256)  # Output: 256 x 7 x 7

        # Attention mechanism (channel attention)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_attention = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 256),
            nn.Sigmoid()
        )

        # Final classification layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        out = self.res_block1(out)
        out = self.res_block2(out)

        # Inception module
        out = self.inception(out)

        # Channel attention
        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b, c)
        y = self.fc_attention(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)

        # Global average pooling and classification
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)

        return out
