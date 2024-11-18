import torch
import torch.nn as nn
import torch.nn.functional as F
from mods.bricks import BasicConv2d, CBAM


class SFI(nn.Module):
    def __init__(self, channel):
        super(SFI, self).__init__()
        self.shared_conv = nn.Conv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.conbine_conv = BasicConv2d(3 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.convy1 = BasicConv2d(channel, channel, 3, 1, 1)
        self.convy2 = BasicConv2d(channel, channel, 3, 1, 1)
        
        self.conv_small = BasicConv2d(channel, channel, 3, 1, 1)
        self.conv_small = BasicConv2d(channel, channel, 3, 1, 1)
        self.conv_large = BasicConv2d(channel, channel, 3, 1, 2, dilation=2)
        self.conv_large = BasicConv2d(channel, channel, 3, 1, 2, dilation=2)

        self.conv_FX1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_FXs = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_FXl = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_FM = nn.Conv2d(3 * channel, channel, kernel_size=3, stride=1, padding=1)

        # Reconstructing the convolutional layer
        self.bn = nn.BatchNorm2d(channel, affine=True)
        self.conva = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.convb = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.convc = BasicConv2d(3 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.convd = BasicConv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.conve = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.convg = BasicConv2d(3 * channel, channel, kernel_size=3, stride=1, padding=1)

        self.cbam = CBAM(channel) 
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(channel, affine=True)
        )

    def forward(self, x, y, z, e):
        # Feature Selection
        X1 = self.bn(self.convc(torch.cat([x, y, z], dim=1)) + x + y)
        X2 = self.conva(x) + x
        X3 = self.convb(y) + y
        X4 = self.mlp_shared(e)

        # Small and large dilated convolutions X2 and X3
        X2_small = self.conv_small(X2)
        X3_large = self.conv_large(X3)
        X2_large = self.conv_large(X2)
        X3_small = self.conv_small(X3)

        # FPN
        M1 = self.conv_FX1(X1)
        M2 = self.conv_FXs(X2_small) + M1
        M3 = self.conv_FXl(X3_large) + M2
        M = self.conv_FM(torch.cat([M1, M2, M3], dim = 1))
        N1 = self.conv_FX1(X1)
        N2 = self.conv_FXl(X2_large) + N1
        N3 = self.conv_FXs(X3_small) + N2
        N = self.conv_FM(torch.cat([N1, N2, N3], dim = 1))

        # Mutate
        Y1 = self.convy1(M * X4)
        Y2 = self.convy2(N * X4)

        # merge
        Y = self.conbine_conv(torch.cat([Y1+self.cbam(Y1), Y2+self.cbam(Y2), X4+self.cbam(X4)], dim=1))

        # output
        Z = Y * (z + 1)
        out = self.convd(self.conve(Z))

        return out