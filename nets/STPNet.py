# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        # skip_x_att = self.pixModule(skip_x)
        # x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        x = torch.cat([skip_x, up], dim=1)
        return self.nConvs(x)


class STPNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=224, vis=False): # ECA:80
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)  # 224->112
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels+1, in_channels* 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2+1, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4+1, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8+1, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        
        self.fc_text1 = nn.Linear(64*224*224, 256)
        self.fc_text2 = nn.Linear(128*112*112, 256)
        self.fc_text3 = nn.Linear(256*56*56, 256)
        self.fc_text4 = nn.Linear(512*28*28, 256)
        self.fc_text5 = nn.Linear(512*14*14, 256)
        
        self.up_text1 = nn.Linear(256*28*28, 256)
        self.up_text2 = nn.Linear(128*56*56, 256)
        self.up_text3 = nn.Linear(64*112*112, 256)
        self.up_text4 = nn.Linear(64*224*224, 256)

    def forward(self, x, text):
        x = x.float()
        text = (text/32).float()
        x1 = self.inc(x)
        y1 = self.downVit(x1, x1)
        
        x2 = self.down1(torch.cat((x1,text),dim=1)) 
        y2 = self.downVit1(x2, y1)
        text = text[:,:,0:112,0:112]
        x3 = self.down2(torch.cat((x2,text),dim=1))
        y3 = self.downVit2(x3, y2) 
        text = text[:,:,0:56,0:56]
        x4 = self.down3(torch.cat((x3,text),dim=1)) 
        y4 = self.downVit3(x4, y3) 
        text = text[:,:,0:28,0:28]
        x5 = self.down4(torch.cat((x4,text),dim=1)) 
        
        x_text5 = x5.view(-1, 512*14*14)
        x_text5 = self.fc_text5(x_text5)
        
        y4 = self.upVit3(y4, y4, True) 
        y3 = self.upVit2(y3, y4, True)
        y2 = self.upVit1(y2, y3, True)
        y1 = self.upVit(y1, y2, True)
        x1 = self.reconstruct1(y1)+x1
        x2 = self.reconstruct2(y2)+x2
        x3 = self.reconstruct3(y3)+x3
        x4 = self.reconstruct4(y4)+x4
        
        x = self.up4(x5, x4) # [16, 256, 28, 28]
        x_text1 = x.view(-1, 256*28*28)
        x = self.up3(x, x3) # [16, 128, 56, 56]
        x_text2 = x.view(-1, 128*56*56)
        x = self.up2(x, x2) # [16, 64, 112, 112]
        x_text3 = x.view(-1, 64*112*112)
        x = self.up1(x, x1) # [16, 64, 224, 224]
        x_text4 = x.view(-1, 64*224*224)
        
        x_text1 = self.up_text1(x_text1)
        x_text2 = self.up_text2(x_text2)
        x_text3 = self.up_text3(x_text3)
        x_text4 = self.up_text4(x_text4)
        
        x_text = [x_text1, x_text2, x_text3, x_text4, x_text5]
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))  # [60,1,112,112]
        else:
            logits = self.outc(x)  # if nusing BCEWithLogitsLoss or class>1
        return logits, x_text
    