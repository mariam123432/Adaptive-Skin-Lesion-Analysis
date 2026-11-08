"""Independent ISIC 2018 experimental implementation; Ali & Ragb (2021) inspired."""
from pathlib import Path
import json, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

IMAGENET_MEAN=torch.tensor([.485,.456,.406],dtype=torch.float32).view(3,1,1)
IMAGENET_STD=torch.tensor([.229,.224,.225],dtype=torch.float32).view(3,1,1)
def normalize_tensor(x):return (x-IMAGENET_MEAN)/IMAGENET_STD

def shades_of_gray(rgb, p=6, eps=1e-6):
    """Shades of Gray illumination normalization, p=6; scale to preserve typical luminance."""
    x=np.asarray(rgb,dtype=np.float32)/255.0
    norms=(np.mean(np.power(np.maximum(x,eps),p),axis=(0,1)))**(1./p)
    gains=np.mean(norms)/(norms+eps)
    y=np.clip(x*gains[None,None,:],0,1)
    return np.uint8(np.round(y*255))
