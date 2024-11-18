import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
from mods.bricks import BasicConv2d
from mods.DCR import DCR
from mods.DGT import DGT, MSDT
from mods.AGF import AGF, MSAG 
import mods.SFI as SFI


class Network(nn.Module):
    def __init__(self, channel=64):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b2()
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1= BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        
        self.DCR = DCR(channel)
        self.MSDT = MSDT(channel)
        self.MSAG = MSAG(channel)
        
        self.DGT1 = DGT(64, channel, 16)
        self.DGT2 = DGT(128, channel, 16)
        self.DGT3 = DGT(320, channel, 16)
        self.DGT4 = DGT(512, channel, 16)

        self.MSDT = MSDT(channel)
        
        self.AGF1 = AGF(64, channel)
        self.AGF2 = AGF(128, channel)
        self.AGF3 = AGF(320, channel)
        self.AGF4 = AGF(512, channel)

        self.MSAG = MSAG(channel)
        
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.SFI1 = SFI(channel)
        self.SFI2 = SFI(channel)
        self.SFI3 = SFI(channel)
        self.SFI4 = SFI(channel)

        self.linearr1 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
        fa1 = self.Translayer1_1(x1)
        fa2 = self.Translayer2_1(x2)#[1, 64, 44, 44]
        fa3 = self.Translayer3_1(x3)
        fa4 = self.Translayer4_1(x4)

        fa, fa_out = self.DCR(fa1, fa2, fa3, fa4)
        fa_out = F.interpolate(fa_out, size=image_shape, mode='bilinear')

        fs1 = self.DGT1(x1) # 18, 64, 88, 88
        fs2 = self.DGT2(x2) # 44*44
        fs3 = self.DGT3(x3) # 22*22
        fs4 = self.DGT4(x4) # 11*11

        fs, fs_out = self.MSDT(fs1, fs2, fs3, fs4)
        fs_out = F.interpolate(fs_out, size=image_shape, mode='bilinear')
        fs22 = F.interpolate(fs2, scale_factor=2, mode='bilinear')
        fs32 = F.interpolate(fs3, scale_factor=4, mode='bilinear')
        fs42 = F.interpolate(fs4, scale_factor=8, mode='bilinear')

        fg1 = self.AGF1(x1)
        fg2 = self.AGF2(x2)
        fg3 = self.AGF3(x3)
        fg4 = self.AGF4(x4)

        fg, fg_out = self.MSAG(fg1, fg2, fg3, fg4)
        fg_out = F.interpolate(fg_out, size=image_shape, mode='bilinear')
        fg22 = F.interpolate(fg2, scale_factor=2, mode='bilinear')
        fg32 = F.interpolate(fg3, scale_factor=4, mode='bilinear')
        fg42 = F.interpolate(fg4, scale_factor=8, mode='bilinear')
        
        fm4 = self.SFI4(fg42, fs42, fs, fa)
        fm3 = self.SFI3(fg32, fs32, fm4, fa)
        fm2 = self.SFI2(fg22, fs22, fm3, fg)
        fm1 = self.SFI1(fg1, fs1, fm2, fg)

        map_4 = self.linearr4(fm4)
        map_3 = self.linearr3(fm3) + map_4
        map_2 = self.linearr2(fm2) + map_3
        map_1 = self.linearr1(fm1) + map_2

        out_1 = F.interpolate(map_1, size=image_shape, mode='bilinear')
        out_2 = F.interpolate(map_2, size=image_shape, mode='bilinear')
        out_3 = F.interpolate(map_3, size=image_shape, mode='bilinear')
        out_4 = F.interpolate(map_4, size=image_shape, mode='bilinear')

        return out_1, out_2, out_3, out_4, fa_out, fg_out, fs_out


if __name__ == '__main__':
    from time import time
    net = Network().cuda()
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352).cuda()
    frame_rate = torch.zeros((1000, 1))
    for i in range(1000):
        torch.cuda.synchronize()
        start = time()
        y = net(dump_x)
        torch.cuda.synchronize()
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        frame_rate[i] = running_frame_rate