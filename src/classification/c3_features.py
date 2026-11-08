"""C3: indices photométriques (RGB uint8 -> caractéristiques numériques).
Les mesures ne constituent pas seules un diagnostic de qualité.
"""
import cv2
import numpy as np

FEATURE_NAMES = [
    "contrast_rms", "contrast_p95_p5", "mean_luma", "median_luma",
    "p05_luma", "p95_luma", "dark_ratio", "bright_ratio",
    "illumination_range", "illumination_cv", "quadrant_mean_range",
]

def _check_rgb(image_rgb):
    a = np.asarray(image_rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError("image_rgb doit être un tableau RGB uint8 HxWx3")
    if min(a.shape[:2]) < 16:
        raise ValueError("Image trop petite (min 16 pixels par dimension)")
    return a

def _luminance(image_rgb, size=256):
    rgb = _check_rgb(image_rgb)
    h, w = rgb.shape[:2]
    scale = min(1.0, size / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (max(16,round(w*scale)), max(16,round(h*scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[:,:,0].astype(np.float32)/255.0

def extract_c3_features(image_rgb):
    """Renvoie un dict stable de 11 mesures dans [0,1] (sauf illumination_cv)."""
    y = _luminance(image_rgb)
    h,w=y.shape
    # Flou d'échelle liée à la taille : attention aux lésions étendues.
    sigma=max(8.0, min(h,w)/9.0)
    field=cv2.GaussianBlur(y,(0,0),sigmaX=sigma,sigmaY=sigma)
    y0,y1=h//2,w//2
    quadrants=[y[:y0,:y1],y[:y0,y1:],y[y0:,:y1],y[y0:,y1:]]
    p05,p95=np.percentile(y,[5,95])
    feats={
        "contrast_rms":float(y.std()),
        "contrast_p95_p5":float(p95-p05),
        "mean_luma":float(y.mean()),
        "median_luma":float(np.median(y)),
        "p05_luma":float(p05),
        "p95_luma":float(p95),
        "dark_ratio":float(np.mean(y < 0.08)),
        "bright_ratio":float(np.mean(y > 0.96)),
        "illumination_range":float(np.percentile(field,95)-np.percentile(field,5)),
        "illumination_cv":float(field.std()/(field.mean()+1e-6)),
        "quadrant_mean_range":float(max(q.mean() for q in quadrants)-min(q.mean() for q in quadrants)),
    }
    return feats
