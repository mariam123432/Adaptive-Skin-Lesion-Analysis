"""C4 conservative preview corrections. Only run with explicit apply=True.
Denoising and sharpening may erase or invent diagnostically relevant details.
"""
import cv2
import numpy as np

def preview_c4(image_rgb, prediction, apply=False, denoise_strength=5.0, sharpen_amount=0.30):
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB image")
    result = image_rgb.copy()
    steps = []
    if apply:
        if prediction.get("C4_2_noise", False):
            # Bilateral acts on native resolution, not arbitrary model enhancement.
            result = cv2.bilateralFilter(result, d=5, sigmaColor=denoise_strength * 4, sigmaSpace=3)
            steps.append("bilateral_denoise")
        if prediction.get("C4_1_blur", False):
            blur = cv2.GaussianBlur(result, (0, 0), sigmaX=1.0)
            result = cv2.addWeighted(result, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)
            steps.append("mild_unsharp_mask")
    delta = cv2.absdiff(result, image_rgb)
    return {"image": result, "applied": bool(steps), "steps": steps,
            "mean_abs_change": float(delta.mean()),
            "changed_pixel_ratio": float(np.any(delta != 0, axis=2).mean())}
