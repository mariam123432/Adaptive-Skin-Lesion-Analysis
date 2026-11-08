"""Independent ISIC 2018 experimental implementation; Ali & Ragb (2021) inspired."""
from pathlib import Path
import json, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

def _lbp8(gray):
    """8-neighbor binary pattern at radius 1 (simple research implementation)."""
    base=gray[1:-1,1:-1]
    shifts=[gray[:-2,:-2],gray[:-2,1:-1],gray[:-2,2:],gray[1:-1,2:],gray[2:,2:],gray[2:,1:-1],gray[2:,:-2],gray[1:-1,:-2]]
    out=np.zeros_like(base,dtype=np.uint8)
    for k,s in enumerate(shifts):out|=((s>=base).astype(np.uint8)<<k)
    return out


def handcrafted200(rgb,mask):
    """Explicit 200-dimensional adapted handcrafted descriptor (NOT exact paper descriptor).
    3 x (16 lesion-color hist + 16 lesion-gradient hist + 16 lesion-LBP hist)
    + 3 x 16 background-color hist + 8 mask geometry = 200.
    """
    im=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)
    m=cv2.resize((mask>0).astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST).astype(bool)
    if m.sum()<4: m[112,112]=True
    feat=[]
    for c in range(3):
        plane=im[:,:,c];gray=plane.astype(np.float32)/255
        gy,gx=np.gradient(gray);g=np.hypot(gx,gy)
        lbp=_lbp8(plane)
        for values,bin_edges in [(plane[m],np.linspace(0,256,17)),(np.clip(g[m],0,1),np.linspace(0,1.000001,17)),(lbp[m[1:-1,1:-1]],np.linspace(0,256,17))]:
            h=np.histogram(values,bins=bin_edges)[0].astype(np.float32)
            feat.extend((h/(h.sum()+1e-8)).tolist())
    for c in range(3):
        vals=im[:,:,c][~m]
        h=np.histogram(vals,bins=np.linspace(0,256,17))[0].astype(np.float32)
        feat.extend((h/(h.sum()+1e-8)).tolist())
    yy,xx=np.where(m)
    a=m.mean();cx=xx.mean()/224;cy=yy.mean()/224
    h=(yy.max()-yy.min()+1)/224;w=(xx.max()-xx.min()+1)/224
    perimeter=cv2.arcLength(max(cv2.findContours((m.astype(np.uint8)*255),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea),True) if m.sum() else 0
    perimeter_norm=perimeter/(2*(224+224))
    elongation=max(h,w)/(min(h,w)+1e-6)
    bbox_area=h*w
    feat.extend([a,cx,cy,h,w,perimeter_norm,elongation,bbox_area])
    arr=np.asarray(feat,dtype=np.float32)
    assert arr.shape==(200,),arr.shape
    return arr

