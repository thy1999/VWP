import os, sys, json
sys.path.append(os.getcwd())
import numpy as np
import torch
from torch import nn
from PIL import Image
import clip
from tqdm import tqdm
import argparse
from pathlib import Path
from .transforms_img_embedding import _transform
from pdb import set_trace as stop

class ExtractFeatureModel(nn.Module):
    ### 捕捉图片编码信息，获取图片特征向量的模型
    def __init__(self,img_encoder):
        super(ExtractFeatureModel, self).__init__()
        ### 定义图片编码层
        img_encoder.attnpool = nn.Identity()
        self.img_encoder = img_encoder
        #self.img_encoder.cuda().eval()

    @torch.no_grad()
    def forward(self, x):
        x = self.img_encoder(x)
        if (len(x.shape)==1):
            x = x.unsqueeze(0)
        return x


class Img_Embedding(nn.Module):
    #RN50x64 ViT-L/14
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    def __init__(self,clip_version="ViT-L/14",img_resolution=224,img_data_dir='/public/home/dzhang/pyProject/hytian/NNER_AS_PARSING_ADD_PIC_WEIBO/weibo_dataset'):
        super(Img_Embedding, self).__init__()
        print("正在使用%s模型对图片信息进行编码！！"%(clip_version))
        clip_model, _ = clip.load(clip_version, device='cpu')  
        self.img_embedding = ExtractFeatureModel(clip_model.visual)
        # img_resolution = clip_model.visual.conv1.weight.shape[0]
        self.img_preprocessor = _transform(img_resolution)
        self.img_data_dir = img_data_dir
    
    def forward(self, img_paths_list):
        clip_feats_list = []
        for img_paths in img_paths_list:
            img_list = []
            for img_path in img_paths:
                img_path = os.path.join(self.img_data_dir, img_path)
                img = Image.open(img_path)
                img = self.img_preprocessor(img).cuda()
                # img.requires_grad = True
                img_list.append(img)
            comb_img = torch.stack(img_list,dim=0)
            # comb_img.requires_grad = True
            # clip_feats = self.img_embedding(comb_img).squeeze(dim=0)
            clip_feats = self.img_embedding(comb_img)
            clip_feats_list.append(clip_feats)
        max_length = max([tensor.shape[0] for tensor in clip_feats_list])
        batch_clip_feats_padding_list = [torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.size(0))) for tensor in clip_feats_list]
        batch_clip_feats = torch.stack(batch_clip_feats_padding_list,dim=0)
        return batch_clip_feats

