import math
from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models
from Fusion_operators.mfbv3 import *
#import ConvMixer
#from Fusion_operators.Bgf import *

import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7,in_channels=3, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        #nn.AdaptiveAvgPool2d((1,1)),
        #nn.Flatten(),
        #nn.Linear(dim, n_classes)
    )
    
class ConvMixer(nn.Module):
    def __init__(self,dim, depth, kernel_size=9, patch_size=7,in_channels=3, n_classes=1000):
        super().__init__()
        self.model = nn.Sequential()
        self.initLayer = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.activation = nn.GELU()
        self.batchNorm = nn.BatchNorm2d(dim)
        self.feature1 = nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim))
        
    def forward(self,x):
        out = self.initLayer(x)
        outactivation = self.activation(out)
        outbatchnorm = self.batchNorm(outactivation)
        outfeature1  = self.feature1(outbatchnorm)
        return outfeature1   

        
    
class ImageCNN(nn.Module):

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c

def normalize_imagenet(x):
    #Normalize input images according to ImageNet standards.
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
 
    def __init__(self, num_classes=512, in_channels=2):
        super().__init__()

        self._model = models.resnet18()
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        features = 0
        for lidar_data in inputs:
            lidar_feature = self._model(lidar_data)
            features += lidar_feature

        return features


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))
        
        # velocity embedding
        #self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor): #, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        
        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]
        
        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0,1,3,4,2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)

        # project velocity to n_embed
        #####velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        
        #######x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        #I did comment this because of errors in the dimensions .................................................
        #x = self.drop(self.pos_emb + token_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = token_embeddings
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        x = x.view(bz, (self.config.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings

        image_tensor_out = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.config.n_views*self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        
        return image_tensor_out, lidar_tensor_out


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))    
        self.config = config
        if self.config.use_convMixEncoder:
            self.image_encoderf1 = ConvMixer(64,1,4,4,in_channels=3)
            self.image_encoderf2 = ConvMixer(128,1,8,8,in_channels=3)
            self.image_encoderf3 = ConvMixer(256,1,16,16,in_channels=3)
            pytorch_total_params = sum(p.numel() for p in self.image_encoderf1.parameters() if p.requires_grad)
            self.lidar_encoderf1 = ConvMixer(64,1,4,4, in_channels=18)
            self.lidar_encoderf2 = ConvMixer(128,1,8,8, in_channels=18)
            self.lidar_encoderf3 = ConvMixer(256,1,16,16, in_channels=18)
            pytorch_total_params2 = sum(p.numel() for p in self.lidar_encoderf1.parameters() if p.requires_grad)
        else:    
            self.image_encoder = ImageCNN(512, normalize=True)
            pytorch_total_params3 = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
            self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=18)
            pytorch_total_params4 = sum(p.numel() for p in self.lidar_encoder.parameters() if p.requires_grad)
        
        self.transformer1 = GPT(n_embd=64,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer2 = GPT(n_embd=128,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)

        
    def forward(self, input, mode = 'fuse'):#, velocity):
 
        if mode == 'fuse': 
            image_list, lidar_list = input
            #if self.image_encoder.normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]
            image_list = torch.stack(image_list)
            bz, _, h, w = lidar_list.shape
            img_channel = image_list.shape[1]
            lidar_channel = lidar_list.shape[1]
            self.config.n_views = 1#len(image_list) // self.config.seq_len

            image_tensor = image_list #torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
            lidar_tensor = lidar_list #torch.stack(lidar_list, dim=1).view(bz * self.config.seq_len, lidar_channel, h, w)

            if self.config.use_convMixEncoder:
                image_features = self.image_encoderf1(image_tensor)
                lidar_features = self.lidar_encoderf1(lidar_tensor)
            else:                    
                image_features = self.image_encoder.features.conv1(image_tensor)
                image_features = self.image_encoder.features.bn1(image_features)
                image_features = self.image_encoder.features.relu(image_features)
                image_features = self.image_encoder.features.maxpool(image_features)

                lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
                lidar_features = self.lidar_encoder._model.bn1(lidar_features)
                lidar_features = self.lidar_encoder._model.relu(lidar_features)
                lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

                image_features = self.image_encoder.features.layer1(image_features)
                lidar_features = self.lidar_encoder._model.layer1(lidar_features)
            # fusion at (B, 64, 64, 64)
            image_embd_layer1 = self.avgpool(image_features)
            lidar_embd_layer1 = self.avgpool(lidar_features)
            image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1)#, velocity)
            #image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=(6,20), mode='bilinear')
            #lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=(6,20), mode='bilinear')
            image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=(8,8), mode='bilinear')
            lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=(8,8), mode='bilinear')
            image_features = image_features + image_features_layer1
            lidar_features = lidar_features + lidar_features_layer1

            if self.config.use_convMixEncoder:
                image_features = self.image_encoderf2(image_tensor)
                lidar_features = self.lidar_encoderf2(lidar_tensor)
            else:
                image_features = self.image_encoder.features.layer2(image_features)
                lidar_features = self.lidar_encoder._model.layer2(lidar_features)
            # fusion at (B, 128, 32, 32)
            image_embd_layer2 = self.avgpool(image_features)
            lidar_embd_layer2 = self.avgpool(lidar_features)
            image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2)#, velocity)
            #image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=(3,10), mode='bilinear')
            #lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=(3,10), mode='bilinear')
            image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=(4,4), mode='bilinear')
            lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=(4,4), mode='bilinear')
            image_features = image_features + image_features_layer2
            lidar_features = lidar_features + lidar_features_layer2

            if self.config.use_convMixEncoder:
                image_features = self.image_encoderf3(image_tensor)
                lidar_features = self.lidar_encoderf3(lidar_tensor)
            else:
                image_features = self.image_encoder.features.layer3(image_features)
                lidar_features = self.lidar_encoder._model.layer3(lidar_features)
            # fusion at (B, 256, 16, 16)
            # image_embd_layer3 = self.avgpool(image_features)
            # lidar_embd_layer3 = self.avgpool(lidar_features)
            # image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3)#, velocity)
            # image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear')
            # lidar_features_layer3 = F.interpolate(lidar_features_layer3, scale_factor=2, mode='bilinear')
            # image_features = image_features + image_features_layer3
            # lidar_features = lidar_features + lidar_features_layer3

            # image_features = self.image_encoder.features.layer4(image_features)
            # lidar_features = self.lidar_encoder._model.layer4(lidar_features)
            # # fusion at (B, 512, 8, 8)
            # image_embd_layer4 = self.avgpool(image_features)
            # lidar_embd_layer4 = self.avgpool(lidar_features)
            # image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4)#, velocity)
            # image_features = image_features + image_features_layer4
            # lidar_features = lidar_features + lidar_features_layer4

            # image_features = self.image_encoder.features.avgpool(image_features)
            # image_features = torch.flatten(image_features, 1)
            # image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)
            # lidar_features = self.lidar_encoder._model.avgpool(lidar_features)
            # lidar_features = torch.flatten(lidar_features, 1)
            # lidar_features = lidar_features.view(bz, self.config.seq_len, -1)

            # fused_features = torch.cat([image_features, lidar_features], dim=1)
            # fused_features = torch.sum(fused_features, dim=1)
            #fused_featuremaps = torch.cat([image_features, lidar_features], dim=1)

            #####     FUSION PART   #####

            #summation
            #fused_featuremaps = torch.add(image_features,lidar_features)

            #MFB
            
            fusion=Fusion_ops()
            fused_featuremaps= fusion.MFBv3(image_features.cpu(),lidar_features.cpu(),0)
            a=2
            #BGF
            #fused_featuremaps=

        elif mode == 'rgb' or mode == 'bev':
            if mode == 'rgb':
                data_list = input[0]
                encoder = self.image_encoder.features
                if self.image_encoder.normalize and mode == 'rgb':
                    data_list = [normalize_imagenet(image_input) for image_input in data_list]
                    data_list = torch.stack(data_list)
            else:
                data_list = input[1]    
                encoder = self.lidar_encoder._model    
            bz, _, h, w = data_list.shape
            img_channel = data_list.shape[1]
            self.config.n_views = 1
            data_features =encoder.conv1(data_list)
            data_features = encoder.bn1(data_features)
            data_features = encoder.relu(data_features)
            data_features = encoder.maxpool(data_features)
            data_features = encoder.layer1(data_features)
            # fusion at (B, 64, 64, 64)
            data_features = encoder.layer2(data_features)
            # fusion at (B, 128, 32, 32)
            data_features = encoder.layer3(data_features)
            fused_featuremaps = data_features


        return fused_featuremaps



class TransFuser(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.out_channels = 512
        self.encoder = Encoder(config).to(self.device)
        self.output = self.encoder


    def forward(self, input):# image_list, lidar_list): 
        #image_list, lidar_list = input

        #### RGB mode #####
        #fused_features = self.encoder(input,mode='rgb')#(image_list, lidar_list)        

        #### Fusion Mode   ###
        fused_features = self.encoder(input)#(image_list, lidar_list)
        #fused_features = self.encoder(input)
        return fused_features


