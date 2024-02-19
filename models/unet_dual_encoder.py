# Load pretrained 2D UNet and modify with temporal attention
import os
import json

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import einsum
import torch.utils.checkpoint
from einops import rearrange


import math

from diffusers import AutoencoderKL
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

def get_unet(pretrained_model_name_or_path, revision=None, add_channels=1):
    # 判断是否是自建模型
    load_pth = os.path.exists(os.path.join(pretrained_model_name_or_path, "unet", "unet.pth"))
    if load_pth:
        # 加载json文件
        with open(os.path.join(pretrained_model_name_or_path, "unet", "config.json"), "r") as f:
            config = json.load(f)
        # 初始化模型结构
        unet = UNet2DConditionModel(**config)
    else:
        # Load pretrained UNet layers
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision
        )

    # Modify input layer to have 1 additional input channels (spine)
    weights = unet.conv_in.weight.clone()
    # get original channels
    ori_channels = weights.shape[1]
    print('ori_channels:', ori_channels)
    # get kernel_size and padding of the original conv_in layer
    ori_kernel_size = unet.conv_in.kernel_size
    print('ori_kernel_size:', ori_kernel_size)
    ori_padding = unet.conv_in.padding
    print('ori_padding:', ori_padding)
    unet.conv_in = nn.Conv2d(ori_channels + add_channels, weights.shape[0], kernel_size=ori_kernel_size, padding=ori_padding) # input noise + spine
    with torch.no_grad():
        unet.conv_in.weight[:, :ori_channels] = weights # original weights
        unet.conv_in.weight[:, ori_channels:] = torch.zeros_like(unet.conv_in.weight[:, ori_channels:]) # new weights initialized to zero

    if load_pth:
        # 加载预训练模型
        unet_chkpt = os.path.join(pretrained_model_name_or_path, "unet", "unet.pth")
        unet_state_dict = torch.load(unet_chkpt, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in unet_state_dict.items():
            name = k if k in unet.state_dict() else k[7:] if k[:7] == 'module.' else 
            new_state_dict[name] = v
        unet.load_state_dict(new_state_dict)

    return unet

def load_text_encoder(pretrained_model_name_or_path, revision=None):
    # 加载文本编码器
    # 加载CLIP的文本模型和分词器，用于将文本转换为嵌入向量
    print(f"Loading text encoder")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, 
                                                # torch_dtype=torch.float16,
                                                subfolder = "text_encoder",
                                                revision=revision).cuda()

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,
                                                subfolder = "tokenizer",
                                                revision=revision)
    print("Text encoder loaded.")
    
    return text_encoder, tokenizer

def getLatent_model(pretrained_model_name_or_path, revision=None):
    # 加载VAE模型
    # 加载VAE模型用于编码和解码图像到隐空间
    print("Loading VAE model")
    load_pth = os.path.exists(os.path.join(pretrained_model_name_or_path, "vae", "vae.pth"))
    if load_pth:
        # 加载json文件
        with open(os.path.join(pretrained_model_name_or_path, "vae", "config.json"), "r") as f:
            config = json.load(f)
        # 初始化模型结构
        vae = AutoencoderKL(**config)
        vae_chkpt = os.path.join(pretrained_model_name_or_path, "vae", "vae.pth")
        vae_state_dict = torch.load(vae_chkpt, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in vae_state_dict.items():
            name = k if k in vae.state_dict() else k[7:] if k[:7] == 'module.' else k
            new_state_dict[name] = v
        vae.load_state_dict(new_state_dict)

    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, 
                                            # torch_dtype=torch.float16,
                                            subfolder = "vae",
                                            revision=revision).cuda()
    print("VAE model loaded.")
    
    return vae
        

class Embedding_Adapter(nn.Module):
    def __init__(self, clip_dim=1024, vae_encoded_dim=1024, compressed_vae_dim=128, output_dim=1024, num_vaes=3, chkpt=None):
        super(Embedding_Adapter, self).__init__()
        
        self.num_vaes = num_vaes
        
        self.pool =  nn.MaxPool2d(2)
        
        # 定义压缩VAE维度的层
        self.compress_vae = nn.Sequential(
            nn.Linear(vae_encoded_dim, compressed_vae_dim),
            nn.ReLU()
        )
        
        # 由于维度被压缩，调整vae2clip层以接受所有压缩后的VAE embeddings
        self.vae2clip = nn.Linear(compressed_vae_dim * num_vaes, clip_dim)
        
        # 定义处理CLIP和VAE embeddings拼接后的层
        self.linear1 = nn.Linear(clip_dim, output_dim)
        
        # 初始化权重
        self.linear1.apply(self._init_weights)

        if chkpt is not None:
            adapter_state_dict = torch.load(chkpt)
            new_state_dict = OrderedDict()
            for k, v in adapter_state_dict.items():
                name = k if k in self.state_dict() else k[7:] if k[:7] == 'module.' else k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            print("Adapter loaded.")
    
    def forward(self, clip, vaes):
        assert isinstance(vaes, list) and len(vaes) == self.num_vaes, f"Expected a list of {self.num_vaes} VAE embeddings, but got {len(vaes)}"
        
        # print('clip:', clip.shape)
        # 使用compress_vae压缩每个VAE embedding
        compressed_vaes = []
        for vae in vaes:
            vae = self.pool(vae) # 1 4 64 64 -> 1 4 32 32
            vae = rearrange(vae, 'b c h w -> b c (h w)') # 1 4 32 32 -> 1 4 1024
            compressed_vaes.append(self.compress_vae(vae))
        
        # 拼接所有压缩后的VAE embeddings
        vae_concat = torch.cat(compressed_vaes, dim=2)
        
        # 将拼接后的VAE embeddings映射到目标维度
        vae_encoded = self.vae2clip(vae_concat)
        # print('vae_encoded:', vae_encoded.shape)
        
        # 拼接CLIP和VAE embeddings
        concat = torch.cat((clip, vae_encoded), dim=1)
        # print('concat:', concat.shape)
        
        # 编码拼接后的embeddings
        encoded = self.linear1(concat)
        
        return encoded

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
