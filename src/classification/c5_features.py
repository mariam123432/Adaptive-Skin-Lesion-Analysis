"""C5 features: ruler and dark peripheral corners. No labels are used at inference."""
import cv2
import numpy as np

FEATURE_NAMES = [
    'gray_mean', 'gray_std', 'edge_ratio', 'edge_periphery_ratio',
    'hough_lines_count', 'hough_total_length', 'hough_max_length',
    'hough_peripheral_fraction', 'hough_axis_fraction',
    'corner_mean', 'center_mean', 'corner_center_ratio',
    'dark_corner_fraction', 'dark_border_fraction', 'dark_center_fraction',
    'corner_asymmetry', 'periphery_brightness_std', 'dark_component_ratio',
]

def _resize(rgb, max_side=384):
    h, w = rgb.shape[:2]
    if min(h, w) < 1: raise ValueError('Image vide')
    factor = min(1., max_side / max(h, w))
    if factor == 1.: return rgb
    return cv2.resize(rgb, (max(1, round(w*factor)), max(1, round(h*factor))), interpolation=cv2.INTER_AREA)

def extract_c5_features(image_rgb):
    """Return float32 array in FEATURE_NAMES order. Uses a standardized copy, not the original pixels."""
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError('Expected HxWx3 RGB image')
    im = _resize(np.asarray(image_rgb, dtype=np.uint8))
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    h,w = gray.shape
    y,x = np.ogrid[:h,:w]
    peripheral = (x < .17*w) | (x >= .83*w) | (y < .17*h) | (y >= .83*h)
    corners = ((x < .20*w) | (x >= .80*w)) & ((y < .20*h) | (y >= .80*h))
    center = (x >= .30*w) & (x < .70*w) & (y >= .30*h) & (y < .70*h)
    edges = cv2.Canny(gray, 55, 150)
    segments = cv2.HoughLinesP(edges, 1, np.pi/180., threshold=35,
                               minLineLength=max(12, int(min(h,w)*.11)),
                               maxLineGap=max(4, int(min(h,w)*.025)))
    lines = [] if segments is None else segments.reshape(-1,4)
    lengths=[]; peripheral_lines=0; axis_lines=0
    for x1,y1,x2,y2 in lines:
        dx,dy=x2-x1,y2-y1
        length=float(np.hypot(dx,dy))
        lengths.append(length)
        mx,my=(x1+x2)/2.,(y1+y2)/2.
        if mx < .20*w or mx > .80*w or my < .20*h or my > .80*h:
            peripheral_lines+=1
        angle=abs(float(np.arctan2(dy,dx))) % (np.pi/2)
        if min(angle, np.pi/2-angle) < np.deg2rad(12): axis_lines+=1
    lengths=np.asarray(lengths, dtype=float)
    dark=gray < 45
    means=[]
    for ys,xs in [(slice(0,int(h*.20)),slice(0,int(w*.20))),
                  (slice(0,int(h*.20)),slice(int(w*.80),w)),
                  (slice(int(h*.80),h),slice(0,int(w*.20))),
                  (slice(int(h*.80),h),slice(int(w*.80),w))]:
        patch=gray[ys,xs]
        means.append(float(np.mean(patch)) if patch.size else float(np.mean(gray)))
    dark_u8=(dark & peripheral).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(dark_u8,8)
    max_component=max([int(stats[i,cv2.CC_STAT_AREA]) for i in range(1,n)],default=0)
    features = [
        gray.mean(), gray.std(), np.mean(edges>0), np.mean((edges>0)[peripheral]),
        len(lengths), lengths.sum()/max(h,w), lengths.max()/max(h,w) if len(lengths) else 0.,
        peripheral_lines/max(len(lines),1), axis_lines/max(len(lines),1),
        gray[corners].mean(), gray[center].mean(),
        gray[corners].mean()/(gray[center].mean()+1e-6),
        dark[corners].mean(), dark[peripheral].mean(), dark[center].mean(),
        np.std(means), gray[peripheral].std(), max_component/max(1,int(peripheral.sum()))
    ]
    out=np.asarray(features,dtype=np.float32)
    if not np.all(np.isfinite(out)): raise ValueError('Non-finite C5 features')
    return out
