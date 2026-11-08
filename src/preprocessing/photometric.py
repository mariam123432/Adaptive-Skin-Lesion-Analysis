"""Prétraitements photométriques C3 : prévisualisation / mode sûr par défaut.
Les transformations sont heuristiques : contrôler impact sur segmentation.
"""
import cv2
import numpy as np

def _check(image_rgb):
    a=np.asarray(image_rgb)
    if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8:
        raise ValueError("RGB uint8 HxWx3 attendu")
    return a

def clahe_luminance(image_rgb, clip_limit=1.7, tile_grid_size=(8,8)):
    rgb=_check(image_rgb)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    l,a,b=cv2.split(lab)
    l2=cv2.createCLAHE(clipLimit=float(clip_limit),tileGridSize=tuple(tile_grid_size)).apply(l)
    return cv2.cvtColor(cv2.merge((l2,a,b)),cv2.COLOR_LAB2RGB)

def correct_illumination(image_rgb, sigma_fraction=0.12, max_gain=1.35):
    rgb=_check(image_rgb)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    l,a,b=cv2.split(lab)
    lf=l.astype(np.float32)/255.0
    sigma=max(6.0,min(lf.shape)*sigma_fraction)
    field=cv2.GaussianBlur(lf,(0,0),sigmaX=sigma,sigmaY=sigma)
    # Limiter le facteur de gain ET d'atténuation.
    gain=np.clip(float(np.median(field))/(field+1e-3), 1/max_gain,max_gain)
    lnew=np.clip(lf*gain,0,1)
    out=cv2.cvtColor(cv2.merge(((lnew*255).astype(np.uint8),a,b)),cv2.COLOR_LAB2RGB)
    return out

def adjust_exposure(image_rgb, gamma=1.0):
    rgb=_check(image_rgb)
    if not 0.65 <= gamma <= 1.5:
        raise ValueError("gamma doit rester entre 0.65 et 1.5")
    table=np.clip(255*(np.arange(256,dtype=np.float32)/255.0)**gamma,0,255).astype(np.uint8)
    return cv2.LUT(rgb,table)

def preview_c3(image_rgb, prediction, apply=False):
    """Retourne original inchangé si apply=False, sinon corrections prudentes."""
    rgb=_check(image_rgb)
    out=rgb.copy()
    operations=[]
    if not apply:
        return {"image":out,"applied":False,"operations":operations}
    # Corriger d'abord illumination/exposition, puis éventuellement contraste.
    if prediction.get("C3_2_uneven_illumination",False):
        out=correct_illumination(out);operations.append("illumination")
    if prediction.get("C3_3_underexposed",False) and not prediction.get("C3_3_overexposed",False):
        out=adjust_exposure(out,gamma=0.85);operations.append("gamma_brighten")
    elif prediction.get("C3_3_overexposed",False) and not prediction.get("C3_3_underexposed",False):
        out=adjust_exposure(out,gamma=1.15);operations.append("gamma_darken")
    if prediction.get("C3_1_low_contrast",False):
        # Après illumination et gamma : sur image corrigée, nécessiterait une réévaluation
        # plus approfondie; ici limité à une première expérimentation.
        out=clahe_luminance(out);operations.append("CLAHE")
    return {"image":out,"applied":bool(operations),"operations":operations}

# Interface compatible avec l'ancien placeholder du projet local.
def clean(image_rgb, subproblems=None, apply=False):
    return preview_c3(image_rgb, subproblems or {}, apply=apply)["image"]
