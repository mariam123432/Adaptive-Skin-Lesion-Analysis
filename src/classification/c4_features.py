"""C4 features: native-resolution focus and noise measurements (no resizing).
The output is a *proxy* for image quality; NOT a clinical annotation.
"""
import cv2
import numpy as np

FEATURE_NAMES = [
    "laplacian_variance", "tenengrad", "noise_mad", "wavelet_sigma",
    "mean_luminance", "luminance_std", "width", "height"
]

def _gray(image_rgb):
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image (H,W,3)")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

def extract_c4_features(image_rgb):
    """Return an interpretable dict of focus/noise indicators on original pixels."""
    gray = _gray(image_rgb)
    h, w = gray.shape
    g = gray.astype(np.float32)

    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gradient2 = gx * gx + gy * gy

    # Median residual excludes stronger gradients when estimating noise.
    med = cv2.medianBlur(gray, 3).astype(np.float32)
    residual = g - med
    limit = float(np.percentile(gradient2, 60))
    flat = residual[gradient2 <= limit]
    if flat.size < 100:
        flat = residual.ravel()
    center = float(np.median(flat))
    noise_mad = float(1.4826 * np.median(np.abs(flat - center)))

    # HH Haar coefficients, orthonormal 2x2 high-high detail channel.
    even_h, even_w = h - h % 2, w - w % 2
    if even_h < 2 or even_w < 2:
        wavelet_sigma = 0.0
    else:
        z = g[:even_h, :even_w]
        hh = (z[0::2, 0::2] - z[0::2, 1::2]
              - z[1::2, 0::2] + z[1::2, 1::2]) / 2.0
        wavelet_sigma = float(np.median(np.abs(hh - np.median(hh))) / 0.67448975)

    out = {
        "laplacian_variance": float(np.var(lap)),
        "tenengrad": float(np.mean(gradient2)),
        "noise_mad": noise_mad,
        "wavelet_sigma": wavelet_sigma,
        "mean_luminance": float(np.mean(g)),
        "luminance_std": float(np.std(g)),
        "width": int(w), "height": int(h),
    }
    return out
