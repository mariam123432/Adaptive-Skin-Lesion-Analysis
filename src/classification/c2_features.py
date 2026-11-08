import cv2
import numpy as np

FEATURE_NAMES = [
    'gray_mean','gray_std','gray_p10','gray_p90',
    'sat_mean','sat_std','val_mean','val_std',
    'edge_ratio','tophat_mean','tophat_p95',
    'circle_count','circle_density','circle_mean_radius',
    'components_count','component_mean_circularity',
    'edge_mean','center_mean','edge_center_diff',
    'edge_std','edge_edge_ratio','border_dark_ratio'
]

def extract_c2_features(image_rgb):
    im = cv2.resize(image_rgb,(256,256),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:,:,1], hsv[:,:,2]
    canny = cv2.Canny(gray,50,130)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    top = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,se)
    circles = cv2.HoughCircles(cv2.GaussianBlur(gray,(7,7),1.5), cv2.HOUGH_GRADIENT,
                 dp=1.4,minDist=15,param1=100,param2=27,minRadius=4,maxRadius=36)
    radii = circles[0,:,2] if circles is not None else np.array([])
    binary = (top > max(12, np.percentile(top,95))).astype(np.uint8)
    contours, _ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    circ=[]
    for ct in contours:
        area=cv2.contourArea(ct); perimeter=cv2.arcLength(ct,True)
        if area>=6 and perimeter>0: circ.append(4*np.pi*area/perimeter**2)
    border = np.zeros((256,256),dtype=bool); k=24
    border[:k,:]=True; border[-k:,:]=True; border[:,:k]=True; border[:,-k:]=True
    inner=~border
    b,cent=gray[border],gray[inner]
    vals = [
      gray.mean(),gray.std(),np.percentile(gray,10),np.percentile(gray,90),
      sat.mean(),sat.std(),val.mean(),val.std(),
      (canny>0).mean(),top.mean(),np.percentile(top,95),
      len(radii),len(radii)/max(gray.size,1),radii.mean() if len(radii) else 0,
      len(circ),np.mean(circ) if circ else 0,
      b.mean(),cent.mean(),b.mean()-cent.mean(),
      b.std(),(canny[border]>0).mean(),(b<45).mean()
    ]
    return np.asarray(vals,dtype=np.float32)
