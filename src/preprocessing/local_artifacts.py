import cv2
import numpy as np
from src.classification.c2_reflection_detector import detect_reflections

def candidate_gel_masks(image_rgb):
    h,w=image_rgb.shape[:2]
    gray=cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)
    scale=min(1.,512/max(h,w))
    sw,sh=max(1,int(w*scale)),max(1,int(h*scale))
    small=cv2.resize(gray,(sw,sh),interpolation=cv2.INTER_AREA)
    smooth=cv2.GaussianBlur(small,(7,7),1.5)
    circles=cv2.HoughCircles(smooth,cv2.HOUGH_GRADIENT,dp=1.4,minDist=15,
                  param1=100,param2=28,minRadius=4,maxRadius=32)
    bubble=np.zeros_like(small,dtype=np.uint8)
    if circles is not None:
        for x,y,r in circles[0]:
            x,y,r=int(x),int(y),int(r)
            cv2.circle(bubble,(x,y),r,255,thickness=2)
    # Approximation faible : zones périphériques de fort contraste. Ne pas inpaint automatiquement.
    grad=cv2.morphologyEx(small,cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    strip=max(6,round(.08*min(sw,sh)))
    border=np.zeros_like(small,dtype=np.uint8)
    border[:strip,:]=1; border[-strip:,:]=1; border[:,:strip]=1; border[:,-strip:]=1
    gel_border=(((grad>np.percentile(grad,92))&(border==1)).astype(np.uint8)*255)
    return { 'gel_bubble':cv2.resize(bubble,(w,h),interpolation=cv2.INTER_NEAREST),
             'gel_border':cv2.resize(gel_border,(w,h),interpolation=cv2.INTER_NEAREST)}

def clean(image_rgb, detection, apply=False, max_area_ratio=.01):
    # Préparer les masques selon la sous-classe prédite
    masks=candidate_gel_masks(image_rgb)
    combined=np.zeros(image_rgb.shape[:2],np.uint8)
    if detection['gel']['gel_bubble']['detected']:
        combined=cv2.bitwise_or(combined,masks['gel_bubble'])
    if detection['gel']['gel_border']['detected']:
        combined=cv2.bitwise_or(combined,masks['gel_border'])
    if detection['reflection_detected']:
        combined=cv2.bitwise_or(combined,detection['reflection_mask'])
    ratio=float((combined>0).mean())
    # Une grande correction ou un masque mal localisé doit être bloqué.
    can_apply=ratio>0 and ratio<=max_area_ratio
    cleaned=image_rgb.copy()
    if apply and can_apply:
        cleaned=cv2.inpaint(image_rgb,combined,3,cv2.INPAINT_TELEA)
    return {'image':cleaned,'mask':combined,'area_ratio':ratio,
            'applied':bool(apply and can_apply),
            'review_required':bool(combined.any())}
