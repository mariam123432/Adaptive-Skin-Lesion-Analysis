"""Independent ISIC 2018 experimental implementation; Ali & Ragb (2021) inspired."""
from pathlib import Path
import json, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .preprocessing import normalize_tensor,shades_of_gray

def dice_loss(logits,y,eps=1.):
    p=logits.sigmoid();dims=(1,2,3)
    d=(2*(p*y).sum(dims)+eps)/(p.sum(dims)+y.sum(dims)+eps)
    return 1-d.mean()


def seg_loss(logits,y): return dice_loss(logits,y)


def seg_metrics(logits,y):
    p=(logits.sigmoid()>=.5).float();dims=(1,2,3)
    inter=(p*y).sum(dims);union=p.sum(dims)+y.sum(dims)
    dice=(2*inter+1e-6)/(union+1e-6)
    iou=(inter+1e-6)/(union-inter+1e-6)
    return float(dice.mean()),float(iou.mean())


def make_segmenter(arch, pretrained=True):
    """Désactiver pretrained pour charger un checkpoint complet sans téléchargement."""
    import segmentation_models_pytorch as smp
    weights = 'imagenet' if pretrained else None
    if arch=='vgg19_unet': return smp.Unet(encoder_name='vgg19',encoder_weights=weights,in_channels=3,classes=1)
    if arch=='deeplabv3plus': return smp.DeepLabV3Plus(encoder_name='resnet50',encoder_weights=weights,in_channels=3,classes=1)
    raise ValueError(arch)


def train_seg(model,train_dl,val_dl,device,output,epochs=12,lr=1e-4):
    model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    amp_enabled=device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    best=-1;history=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for x,y in train_dl:
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp_enabled):
                pred=model(x);loss=seg_loss(pred,y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss.item()))
        model.eval();ds=[];ious=[]
        with torch.no_grad():
            for x,y in val_dl:
                x=x.to(device);y=y.to(device)
                pred=model(x);d,j=seg_metrics(pred,y);ds.append(d*len(x));ious.append(j*len(x))
        denom=len(val_dl.dataset);vd=sum(ds)/denom;vi=sum(ious)/denom
        history.append({'epoch':epoch,'train_dice_loss':float(np.mean(losses)),'val_dice':vd,'val_iou':vi})
        print(f'Epoch {epoch}: loss={np.mean(losses):.4f}; val_Dice={vd:.4f}; val_IoU={vi:.4f}',flush=True)
        if vd>best:
            best=vd;torch.save({'state_dict':model.state_dict(),'val_dice':vd,'epoch':epoch},output)
        with open(output.with_suffix('.history.json'),'w') as f: json.dump(history,f,indent=2)
    model.load_state_dict(torch.load(output,map_location=device,weights_only=False)['state_dict']);return history


def ensemble_mask(rgb,models,device,size=224,return_native=True):
    """Mean of probability matrices, then binary mask; mask returned in input size."""
    original_hw=rgb.shape[:2]
    resized=shades_of_gray(cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA),p=6)
    t=normalize_tensor(torch.from_numpy(resized.transpose(2,0,1).copy()).float()/255)[None].to(device)
    probs=[]
    for model in models:
        model.eval();probs.append(model(t).sigmoid())
    mean=torch.stack(probs).mean(0)[0,0].cpu().numpy()
    if return_native:
        mean=cv2.resize(mean,(original_hw[1],original_hw[0]),interpolation=cv2.INTER_LINEAR)
    return (mean>=.5).astype(np.uint8),mean

