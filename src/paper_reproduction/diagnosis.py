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

class HybridCNN(nn.Module):
    def __init__(self,architecture,feature_dim=200,num_classes=7,pretrained=True):
        super().__init__()
        from torchvision import models
        if architecture=='resnet50':
            self.backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            dim=self.backbone.fc.in_features;self.backbone.fc=nn.Identity()
        elif architecture=='densenet201':
            self.backbone=models.densenet201(weights=models.DenseNet201_Weights.DEFAULT if pretrained else None)
            dim=self.backbone.classifier.in_features;self.backbone.classifier=nn.Identity()
        else: raise ValueError(architecture)
        self.fuse=nn.Sequential(nn.Linear(dim+feature_dim,512),nn.ReLU(),nn.Dropout(.35))
        self.head=nn.Linear(512,num_classes)
    def embedding(self,image,manual):
        x=self.backbone(image)
        return self.fuse(torch.cat((x,manual),dim=1))
    def forward(self,image,manual):return self.head(self.embedding(image,manual))


def train_hybrid(model,train_dl,val_dl,device,output,epochs=6,lr=1e-4):
    model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=lr)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    from sklearn.metrics import balanced_accuracy_score
    best=-1;history=[]
    for ep in range(1,epochs+1):
        model.train();losses=[]
        for x,f,y in train_dl:
            x=x.to(device);f=f.to(device);y=y.to(device)
            opt.zero_grad(set_to_none=True);logits=model(x,f)
            loss=F.cross_entropy(logits,y)
            loss.backward();opt.step();losses.append(loss.item())
        model.eval();yt=[];yp=[]
        with torch.no_grad():
            for x,f,y in val_dl:
                pred=model(x.to(device),f.to(device)).argmax(1).cpu().numpy()
                yt.extend(y.numpy());yp.extend(pred)
        score=balanced_accuracy_score(yt,yp)
        history.append({'epoch':ep,'train_ce':float(np.mean(losses)),'val_balanced_acc':float(score)})
        print(f'Hybrid epoch {ep}: CE={np.mean(losses):.4f}, BACC={score:.4f}',flush=True)
        if score>best:
            best=score;torch.save({'state_dict':model.state_dict(),'epoch':ep,'val_balanced_acc':score},output)
        output.with_suffix('.history.json').write_text(json.dumps(history,indent=2))
    model.load_state_dict(torch.load(output,map_location=device,weights_only=False)['state_dict']);return history


def extract_embeddings(model,dl,device):
    model.eval();xs=[];ys=[]
    for x,f,y in dl:
        e=model.embedding(x.to(device),f.to(device)).cpu().numpy()
        xs.append(e);ys.append(y.numpy())
    return np.concatenate(xs),np.concatenate(ys)

