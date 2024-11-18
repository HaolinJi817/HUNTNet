import torch
import torch.nn as nn
from mods.bricks import BasicConv2d, AttentionBlock


class DCR(nn.Module):
    def __init__(self, channel):
        super(DCR, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Conv Layers
        self.conv_upsample = nn.ModuleList([
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        ])
        
        # Fusion Layers
        self.conv_fusion = nn.ModuleList([
            BasicConv2d(4 * channel, 2 * channel, 3, padding=1),  # 通道从 4 * channel -> 2 * channel
            BasicConv2d(4 * channel, 2 * channel, 3, padding=1),  # 同上
            BasicConv2d(4 * channel, 4 * channel, 3, padding=1)   # 通道保持 4 * channel
        ])

        # Output Layers
        self.conv_output = nn.ModuleList([
            BasicConv2d(4 * channel, 4 * channel, 3, padding=1),  # 输入为 4 * channel
            nn.Conv2d(4 * channel, channel, 1),
            nn.Conv2d(channel, 1, 1)
        ])

        # Attention
        self.attention = nn.ModuleList([
            AttentionBlock(2 * channel),
            AttentionBlock(2 * channel),
            AttentionBlock(2 * channel),
            AttentionBlock(2 * channel),
            AttentionBlock(2 * channel) 
        ])

    def forward(self, x1, x2, x3, x4):
        # Recursion + Attention
        x4_1 = self.attention[0](torch.cat((x4, x4), 1))  # 通道数 = 2 * channel
        x3_1 = self.attention[1](self.conv_upsample[0](self.upsample(x4_1)) + torch.cat((x3, x3), 1))  # 通道数 = 2 * channel
        
        x2_1 = self.attention[2](self.conv_upsample[1](self.upsample(x3_1)) * self.conv_upsample[2](self.upsample(torch.cat((x3, x3), 1))) + torch.cat((x2, x2), 1))  # 通道数 = 2 * channel
        x1_1 = self.attention[3](self.conv_upsample[3](self.upsample(x2_1)) * self.conv_upsample[3](self.upsample(torch.cat((x2, x2), 1))) + torch.cat((x1, x1), 1))  # 通道数 = 2 * channel

        # Recursion + Attention
        x3_2 = self.attention[4](self.conv_fusion[0](torch.cat((x3_1, self.conv_upsample[0](self.upsample(x4_1))), 1)))  # 通道数 = 4 * channel
        x2_2 = self.attention[4](self.conv_fusion[1](torch.cat((x2_1, self.conv_upsample[1](self.upsample(x3_2))), 1)))  # 通道数 = 4 * channel
        x1_2 = self.conv_fusion[2](torch.cat((x1_1, self.conv_upsample[2](self.upsample(x2_2))), 1))  # 通道数 = 4 * channel

        # output
        out = self.conv_output[0](x1_2)
        out1 = self.conv_output[1](out)
        out2 = self.conv_output[2](out1)

        return out1, out2