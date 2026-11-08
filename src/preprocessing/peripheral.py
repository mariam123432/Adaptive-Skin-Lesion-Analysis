"""Experimental candidate masks for C5. Corrections are disabled by default."""
import cv2
import numpy as np


def dark_corner_valid_mask(image_rgb, cutoff=45, periphery_ratio=.24, dilate_px=3):
    """255=valid field, 0=invalid peripheral background; do not discard lesions.

    Mask is a candidate requiring inspection before use in segmentation.
    """
    gray=cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    yy,xx=np.ogrid[:h,:w]
    periphery=(xx < periphery_ratio*w)|(xx >= (1-periphery_ratio)*w)| (yy < periphery_ratio*h)|(yy >= (1-periphery_ratio)*h)
    dark=((gray<cutoff)&periphery).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(dark,8)
    invalid=np.zeros((h,w),np.uint8)
    # Only peripheral components touching actual image edges
    for i in range(1,n):
        x,y,cw,ch,area=stats[i]
        if area < max(12,int(h*w*.00003)):continue
        if x==0 or y==0 or x+cw>=w or y+ch>=h:
            invalid[labels==i]=255
    if dilate_px>0:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate_px+1,2*dilate_px+1))
        invalid=cv2.dilate(invalid,kernel)
    return 255-invalid


def ruler_candidate_mask(image_rgb, min_line_fraction=.12):
    """Candidate peripheral linear structures; NOT a validated ruler segmentation."""
    gray=cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    edges=cv2.Canny(gray,65,145)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=45,
                          minLineLength=max(18,int(min(h,w)*min_line_fraction)),
                          maxLineGap=max(5,int(min(h,w)*.015)))
    mask=np.zeros((h,w),dtype=np.uint8)
    if lines is None:return mask
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        mx,my=(x1+x2)/2.,(y1+y2)/2.
        if mx<.20*w or mx>.80*w or my<.20*h or my>.80*h:
            cv2.line(mask,(x1,y1),(x2,y2),255,thickness=max(2,int(min(h,w)*.003)))
    return mask


def preview_c5(image_rgb,prediction):
    """Return annotations/masks; ALWAYS keep original unchanged."""
    result={'image':image_rgb.copy(),'cleaned':image_rgb.copy(), 'applied':False,
            'valid_field_mask':np.full(image_rgb.shape[:2],255,np.uint8),
            'ruler_candidate_mask':np.zeros(image_rgb.shape[:2],np.uint8)}
    if prediction.get('dark_corner',{}).get('detected',False):
        result['valid_field_mask']=dark_corner_valid_mask(image_rgb)
    if prediction.get('ruler',{}).get('detected',False):
        result['ruler_candidate_mask']=ruler_candidate_mask(image_rgb)
    return result
