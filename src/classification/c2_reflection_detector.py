import cv2
import numpy as np

DEFAULT_CONFIG={
  'v_min':225, 's_max':75, 'tophat_min':16,
  'min_area':8, 'max_region_ratio':0.005, 'max_total_ratio':0.015,
  'tophat_kernel':17,
}

def detect_reflections(image_rgb, config=None):
    cfg={**DEFAULT_CONFIG,**(config or {})}
    hsv=cv2.cvtColor(image_rgb,cv2.COLOR_RGB2HSV)
    gray=cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)
    v,s=hsv[:,:,2],hsv[:,:,1]
    k=cfg['tophat_kernel']; k = k+1 if k%2==0 else k
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    th=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel)
    bright=(v>=cfg['v_min']) & (s<=cfg['s_max']) & (th>=cfg['tophat_min'])
    n,lab,stats,centroids=cv2.connectedComponentsWithStats(bright.astype(np.uint8),8)
    mask=np.zeros_like(gray,dtype=np.uint8)
    max_area=max(1,int(gray.size*cfg['max_region_ratio']))
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA])
        if cfg['min_area']<=area<=max_area:
            mask[lab==i]=255
    ratio=float((mask>0).mean())
    return {'reflection_detected':bool(mask.any()),'mask':mask,'area_ratio':ratio,
            'safe_to_inpaint':bool(mask.any() and ratio<=cfg['max_total_ratio']),
            'tophat':th}
