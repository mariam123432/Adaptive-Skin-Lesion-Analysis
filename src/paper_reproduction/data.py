"""Independent ISIC 2018 experimental implementation; Ali & Ragb (2021) inspired."""
from pathlib import Path
import json, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .settings import CLASSES
from .preprocessing import normalize_tensor, shades_of_gray

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_rgb(path):
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def image_lookup(root):
    files=sorted(Path(root).rglob('*.jpg'))+sorted(Path(root).rglob('*.JPG'))
    return {p.stem:p for p in files}


def mask_lookup(root):
    files=sorted(Path(root).rglob('*.png'))
    return {p.stem.replace('_segmentation',''):p for p in files}


def build_pairs(image_root,mask_root):
    x=image_lookup(image_root); y=mask_lookup(mask_root)
    common=sorted(set(x)&set(y))
    if not common: raise ValueError('Aucune paire image/masque. Vérifier les ZIP et chemins.')
    print('Images:',len(x),'Masques:',len(y),'Paires:',len(common))
    return [{'image_id':i,'image_path':str(x[i]),'mask_path':str(y[i])} for i in common]


class SegDataset(Dataset):
    def __init__(self, rows, size=224, augment=False): self.rows=rows;self.size=size;self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        rgb=read_rgb(row['image_path'])
        mask=cv2.imread(str(row['mask_path']),cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(row['mask_path'])
        rgb=cv2.resize(rgb,(self.size,self.size),interpolation=cv2.INTER_AREA)
        rgb=shades_of_gray(rgb,p=6)
        mask=cv2.resize(mask,(self.size,self.size),interpolation=cv2.INTER_NEAREST)
        if self.augment:
            # Transformations synchronisées de l'image et du masque, seulement Train.
            if random.random()<.5: rgb=np.flip(rgb,axis=1);mask=np.flip(mask,axis=1)
            if random.random()<.5: rgb=np.flip(rgb,axis=0);mask=np.flip(mask,axis=0)
            # Rotation aléatoire dans l'intervalle 0–180 degrés (papier).
            degree=random.uniform(0,180)
            M=cv2.getRotationMatrix2D((self.size/2,self.size/2),degree,1.0)
            rgb=cv2.warpAffine(rgb,M,(self.size,self.size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            mask=cv2.warpAffine(mask,M,(self.size,self.size),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            # Cisaillement optionnel de 0–30 degrés.
            if random.random()<.5:
                k=np.tan(np.deg2rad(random.uniform(-30,30)))
                shear=np.float32([[1,k,-k*self.size/2],[0,1,0]])
                rgb=cv2.warpAffine(rgb,shear,(self.size,self.size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                mask=cv2.warpAffine(mask,shear,(self.size,self.size),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            # Recadrage aléatoire conjoint puis redimensionnement.
            if random.random()<.5:
                crop_size=max(16,int(self.size*random.uniform(.85,1.0)))
                yy=random.randint(0,self.size-crop_size)
                xx=random.randint(0,self.size-crop_size)
                rgb=cv2.resize(rgb[yy:yy+crop_size,xx:xx+crop_size],(self.size,self.size),interpolation=cv2.INTER_LINEAR)
                mask=cv2.resize(mask[yy:yy+crop_size,xx:xx+crop_size],(self.size,self.size),interpolation=cv2.INTER_NEAREST)
        x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()/255
        m=torch.from_numpy(np.ascontiguousarray(mask[None]>127)).float()
        return normalize_tensor(x),m


class DiagnosisDataset(Dataset):
    """Use cached 200 handcrafted features and image RGB; no mask required at inference."""
    def __init__(self,rows,features,size=224):self.rows=rows;self.features=features;self.size=size
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];rgb=read_rgb(row['image_path'])
        rgb=cv2.resize(rgb,(self.size,self.size),interpolation=cv2.INTER_AREA)
        rgb=shades_of_gray(rgb,6)
        x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255
        f=torch.from_numpy(self.features[row['image_id']]).float()
        return normalize_tensor(x),f,torch.tensor(CLASSES.index(row['label']),dtype=torch.long)

