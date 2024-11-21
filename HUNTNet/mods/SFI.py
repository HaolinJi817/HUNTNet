import torch
import torch.nn as nn
import torch.nn.functional as F
from mods.bricks import BasicConv2d, CBAM

class SFI(nn.Module):
    def __init__(self, channel):
        super(SFI, self).__init__()
        # Reduce redundant convolution layers
        self.shared_conv = nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.conbine_conv = BasicConv2d(3 * channel, channel, kernel_size=1, stride=1, padding=0)  # Using 1x1 conv to reduce MACs
        
        # Simplified convolutions with shared weights
        self.conv_base = BasicConv2d(channel, channel, 3, 1, 1)
        self.conv_dilated = BasicConv2d(channel, channel, 3, 1, 2, dilation=2)

        # FPN convolutions (Reduced number of convolution layers)
        self.conv_FX = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_FM = nn.Conv2d(3 * channel, channel, kernel_size=1, stride=1, padding=0)  # Use 1x1 conv to reduce MACs

        # Reconstructing the convolutional layers
        self.bn = nn.BatchNorm2d(channel, affine=True)
        self.conva = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel)  # Depthwise conv
        self.convc = BasicConv2d(3 * channel, channel, kernel_size=1, stride=1, padding=0)  # Using 1x1 conv
        self.convd = BasicConv2d(channel, channel, kernel_size=1, stride=1, padding=0)  # Using 1x1 conv

        # Lightweight attention mechanism
        self.cbam = CBAM(channel)  
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0),  # Use 1x1 conv to reduce MACs
            nn.ReLU(True),
            nn.BatchNorm2d(channel, affine=True)
        )

    def forward(self, x, y, z, e):
        # Feature Selection
        X1 = self.bn(self.convc(torch.cat([x, y, z], dim=1)) + x + y)
        X2 = self.conva(x) + x
        X3 = self.conva(y) + y  # Use the same depthwise conv to reduce parameters
        X4 = self.mlp_shared(e)

        # Small and large dilated convolutions X2 and X3
        X2_small = self.conv_base(X2)
        X3_large = self.conv_dilated(X3)
        X2_large = self.conv_dilated(X2)
        X3_small = self.conv_base(X3)

        # FPN (Reduced number of branches)
        M1 = self.conv_FX(X1)
        M2 = self.conv_FX(X2_small) + M1
        M3 = self.conv_FX(X3_large) + M2
        M = self.conv_FM(torch.cat([M1, M2, M3], dim=1))
        
        N1 = self.conv_FX(X1)
        N2 = self.conv_FX(X2_large) + N1
        N3 = self.conv_FX(X3_small) + N2
        N = self.conv_FM(torch.cat([N1, N2, N3], dim=1))

        # Mutate
        Y1 = self.conv_base(M * X4)
        Y2 = self.conv_base(N * X4)

        # Merge
        Y = self.conbine_conv(torch.cat([Y1 + self.cbam(Y1), Y2 + self.cbam(Y2), X4 + self.cbam(X4)], dim=1))

        # Output
        Z = Y * (z + 1)
        out = self.convd(Z)

        return out
