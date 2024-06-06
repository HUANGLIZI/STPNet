# -*- coding: utf-8 -*-
#79.35
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from .ViT_text import VisionTransformer, Reconstruct #ViT_text
from transformers import BertModel
from .ImageAggr import EncoderImageAggr
from torchvision.models.resnet import resnet101
from .mlp import FC_MLP
import pickle
# from .pixlevel import PixLevelModule


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
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        x = torch.cat([skip_x, up], dim=1)
        return self.nConvs(x)

class MSAtt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_spatial_1 = nn.Conv2d(dim, dim, 3, stride=1, padding=6, groups=dim, dilation=6)
        self.conv_spatial_2 = nn.Conv2d(dim, dim, 3, stride=1, padding=12, groups=dim, dilation=12)
        self.conv_spatial_3 = nn.Conv2d(dim, dim, 3, stride=1, padding=18, groups=dim, dilation=18)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn_1 = self.conv_spatial_1(x)
        attn_2 = self.conv_spatial_2(x)
        attn_3 = self.conv_spatial_3(x)
        attn = attn_1 + attn_2 + attn_3
        attn = self.conv1(attn)

        return u * attn

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True, groups=in_channels)
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dwconv(x)
        weights = self.conv1(x)
        weights = self.sigmoid(weights)
        attended_x = x * weights
        return attended_x

class SSA(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.ReLU(inplace=True)
        self.dwspatialatt=SpatialAttention(d_model)
        self.spatial_gating_unit = MSAtt(d_model)
        self.proj_2 = nn.Conv2d(d_model*2, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        xdw = self.dwspatialatt(x)
        xsg = self.spatial_gating_unit(x)
        x = self.proj_2(torch.cat([xdw,xsg],dim=1))
        x = x + shorcut
        return x

class ImgEncoder(nn.Module):
    def __init__(self, opt = {}, finetune=True):
        super(ImgEncoder, self).__init__()

        self.embed_dim = 768

        self.resnet = resnet101(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = finetune

        self.pool_2x2 = nn.MaxPool2d(2)

        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample_4 = nn.Upsample(scale_factor=4, mode='nearest')

        self.linear = nn.Linear(in_features=512, out_features=self.embed_dim)

    def forward(self, img):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        f4_up = self.up_sample_2(f4)
        higher_feature = torch.cat([f3, f4_up], dim=1)

        return higher_feature
    
class FB(nn.Module):
    def __init__(self,  n_feat, kernel_size, bias=False, act=nn.ReLU(True)):
        super(FB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
class STPNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=224, vis=False): # ECA:80
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.clsdown1 = DownBlock(in_channels, in_channels* 2, nb_Conv=2)
        self.van_cls1 = SSA(d_model=in_channels* 2)
        self.clsdown2 = DownBlock(in_channels*2, in_channels* 4, nb_Conv=2)
        self.van_cls2 = SSA(d_model=in_channels* 4)
        self.clsdown3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.van_cls3 = SSA(d_model=in_channels*8)
        self.norm = nn.LayerNorm(768)
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
        self.text_conv1 = nn.Linear(768, in_channels)
        self.text_conv2 = nn.Linear(768, in_channels * 2)
        self.text_conv3 = nn.Linear(768, in_channels * 4)
        self.text_conv4 = nn.Linear(768, in_channels * 8)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=1, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=64, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=128, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=256, kernel_size=1, scale_factor=(2, 2))

        self.fusion_block4 = nn.Sequential(
                FB(in_channels * 8, 1, act=nn.ReLU(True)))
        self.fusion_cnn4 = nn.Sequential(
                nn.Conv2d(in_channels * 4, in_channels * 4, 3 ,padding=(3 // 2)), nn.ReLU(True), nn.Conv2d(in_channels * 4, in_channels * 4, 3,padding=(3 // 2)))
        self.fusion_block3 = nn.Sequential(
                FB(in_channels * 4, 1, act=nn.ReLU(True)))
        self.fusion_cnn3 = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels * 2, 3,padding=(3 // 2)), nn.ReLU(True), nn.Conv2d(in_channels * 2, in_channels * 2, 3,padding=(3 // 2)))
        self.fusion_block2 = nn.Sequential(
                FB(in_channels * 2, 1, act=nn.ReLU(True)))
        self.fusion_cnn2 = nn.Sequential(
                nn.Conv2d(in_channels * 1, in_channels * 1, 3,padding=(3 // 2)), nn.ReLU(True), nn.Conv2d(in_channels * 1, in_channels * 1, 3,padding=(3 // 2)))
        self.fusion_block1 = nn.Sequential(
                FB(in_channels+1 , 1, act=nn.ReLU(True)))
        self.fusion_cnn1 = nn.Sequential(
                nn.Conv2d(1, 1, 3,padding=(3 // 2)), nn.ReLU(True), nn.Conv2d(1, 1, 3,padding=(3 // 2)))
        
        self.imgencoder = ImgEncoder()
        self.imgaggr = EncoderImageAggr(img_dim=768, embed_size=768)
        self.img_linear1 = FC_MLP(768,64,2,3)
        self.img_linear2 = FC_MLP(768,64,2,2)
        self.img_linear3 = FC_MLP(768,64,8,4)
        self.img_linear4 = FC_MLP(768,64,8,4)

        with open('./Text/BUnilateral.pkl', 'rb') as pkl_file:
            self.Unilateral_emb = pickle.load(pkl_file)
        self.Unilateral_emb = torch.Tensor(self.Unilateral_emb).cuda()
        with open('./Text/num.pkl', 'rb') as pkl_file:
            self.num_emb = pickle.load(pkl_file)
        self.num_emb = torch.Tensor(self.num_emb).cuda()
        with open('./Text/left_loc.pkl', 'rb') as pkl_file:
            self.left_loc_emb = pickle.load(pkl_file)
        self.left_loc_emb = torch.Tensor(self.left_loc_emb).cuda()
        with open('./Text/right_loc.pkl', 'rb') as pkl_file:
            self.right_loc_emb = pickle.load(pkl_file)
        self.right_loc_emb = torch.Tensor(self.right_loc_emb).cuda()
    def l2norm(self,X, dim, eps=1e-8):
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X    
    def compute_logits(self, img_emb, text_emb):
        similarities = torch.matmul(img_emb, text_emb.t())
        return  similarities
    def forward(self, x):
        Unilateral_emb = self.Unilateral_emb.mean(2).mean(1) 
        Unilateral_emb_norm = self.norm(Unilateral_emb)
        num_emb = self.num_emb.mean(2).mean(1) 
        num_emb_norm = self.norm(num_emb)
        left_loc_emb = self.left_loc_emb.mean(2).mean(1) 
        left_loc_emb_norm = self.norm(left_loc_emb)
        right_loc_emb = self.right_loc_emb.mean(2).mean(1) 
        right_loc_emb_norm = self.norm(right_loc_emb)
        x1 = self.inc(x)
        
        higher_feature = self.imgencoder(x)
        
        img_emb=higher_feature.reshape(x.shape[0],768,-1).permute(0,2,1)
        
        image_lengths=img_emb.shape[-1]*torch.ones((img_emb.shape[0])).to(img_emb.device)
        
        img_emb = self.imgaggr(img_emb,image_lengths)
        Unilateral_img_emb = self.img_linear1(img_emb)
        num_img_emb = self.img_linear2(img_emb)
        left_loc_img_emb = self.img_linear3(img_emb)
        right_loc_img_emb = self.img_linear4(img_emb)
        img_emb = self.norm(img_emb)
        


        Unilateral_logits_per_image = self.compute_logits(img_emb,Unilateral_emb_norm)
        Unilateral_max_indices = torch.argmax(Unilateral_logits_per_image, dim=1)

        num_logits_per_image = self.compute_logits(img_emb,num_emb_norm)
        num_max_indices = torch.argmax(num_logits_per_image, dim=1)

        left_loc_logits_per_image = self.compute_logits(img_emb,left_loc_emb_norm)
        left_loc_max_indices = torch.argmax(left_loc_logits_per_image, dim=1)

        right_loc_logits_per_image = self.compute_logits(img_emb,right_loc_emb_norm)
        right_loc_max_indices = torch.argmax(right_loc_logits_per_image, dim=1)
        
        selected_text_emb_trans1 = self.Unilateral_emb.mean(1)[Unilateral_max_indices]
        selected_text_emb_trans2 = torch.stack((self.Unilateral_emb.mean(1)[Unilateral_max_indices],self.num_emb.mean(1)[num_max_indices]))
        selected_text_emb_trans3 = torch.stack((self.Unilateral_emb.mean(1)[Unilateral_max_indices],self.num_emb.mean(1)[num_max_indices],self.left_loc_emb.mean(1)[left_loc_max_indices]))
        selected_text_emb_trans4 = torch.stack((self.Unilateral_emb.mean(1)[Unilateral_max_indices],self.num_emb.mean(1)[num_max_indices],self.left_loc_emb.mean(1)[left_loc_max_indices],self.right_loc_emb.mean(1)[right_loc_max_indices]))

        selected_text_emb_trans2 = selected_text_emb_trans2.mean(0)
        selected_text_emb_trans3 = selected_text_emb_trans3.mean(0)
        selected_text_emb_trans4 = selected_text_emb_trans4.mean(0)
        
        x = x.float()

        text=self.text_conv1(selected_text_emb_trans1) 
        y1 = self.downVit(x1, x1, text)

        text_cnn = text.mean(1).mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,224,224)
        x2 = self.down1(torch.cat((x1,text_cnn),dim=1)) 
        x2 = self.van_cls1(x2)
        
        text=self.text_conv2(selected_text_emb_trans2)
        y2 = self.downVit1(x2, y1, text)
        text_cnn = text.mean(1).mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,112,112)
        x3 = self.down2(torch.cat((x2,text_cnn),dim=1))
        x3 = self.van_cls2(x3)
     
        text=self.text_conv3(selected_text_emb_trans3)
        y3 = self.downVit2(x3, y2, text) 
        text_cnn = text.mean(1).mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,56,56)
        x4 = self.down3(torch.cat((x3,text_cnn),dim=1)) 
        x4 = self.van_cls3(x4)
     
        text=self.text_conv4(selected_text_emb_trans4)
        y4 = self.downVit3(x4, y3, text) 

        text_cnn = text.mean(1).mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,28,28)
        x5 = self.down4(torch.cat((x4,text_cnn),dim=1)) 
        
        
        y4 = self.upVit3(y4, y4, reconstruct=True) 
        y3 = self.upVit2(y3, y4, reconstruct=True)
        y2 = self.upVit1(y2, y3, reconstruct=True)
        y1 = self.upVit(y1, y2, reconstruct=True)
        x1_trans = self.reconstruct1(y1)       
        x2_trans = self.reconstruct2(y2)       
        x3_trans = self.reconstruct3(y3)       
        x4_trans = self.reconstruct4(y4)       
        
        x4_cnn = self.up4(x5, x4) 
        _, x_res = x4_trans, x4_cnn
        f = torch.cat((x4_cnn, x4_trans), 1)
        f = f + self.fusion_block4(f)
        x4_trans, x4_cnn = torch.split(f, x4_cnn.shape[1], 1)
        x = self.fusion_cnn4(x4_cnn) + x_res

        x3_cnn = self.up3(x, x3) 
        _, x_res = x3_trans, x3_cnn
        f = torch.cat((x3_cnn, x3_trans), 1)
        f = f + self.fusion_block3(f)
        x3_trans, x3_cnn = torch.split(f, x3_cnn.shape[1], 1)
        x = self.fusion_cnn3(x3_cnn) + x_res


        x2_cnn = self.up2(x, x2) 
        _, x_res = x2_trans, x2_cnn
        f = torch.cat((x2_cnn, x2_trans), 1)
        f = f + self.fusion_block2(f)
        x2_trans, x2_cnn = torch.split(f, x2_cnn.shape[1], 1)
        x = self.fusion_cnn2(x2_cnn) + x_res

        x1_cnn = self.up1(x, x1) 
        _, x_res = x1_trans, x1_cnn
        f = torch.cat((x1_cnn, x1_trans), 1)
        f = f + self.fusion_block1(f)
        x1_trans, x1_cnn = torch.split(f, x1_cnn.shape[1], 1)
        x = self.fusion_cnn1(x1_cnn) + x_res
        

        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))  
        else:
            logits = self.outc(x)  

        return logits, img_emb, Unilateral_img_emb,num_img_emb,left_loc_img_emb,right_loc_img_emb,Unilateral_emb_norm,num_emb_norm,left_loc_emb_norm,right_loc_emb_norm
    