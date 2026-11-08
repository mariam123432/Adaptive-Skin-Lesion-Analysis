"""C4 rule-based classifier. Thresholds in models/c4_config.json are provisional.
Requires independent visual labels before accuracy/F1 can be reported.
"""
import json
from pathlib import Path
from src.classification.c4_features import extract_c4_features

class C4Detector:
    def __init__(self, project_path):
        path = Path(project_path) / "models" / "c4_config.json"
        with path.open(encoding="utf-8") as f:
            self.config = json.load(f)
        for key in ("blur_laplacian_max", "blur_tenengrad_max", "noise_mad_min", "noise_wavelet_min"):
            if key not in self.config:
                raise KeyError(f"Missing C4 threshold: {key}")

    def predict(self, image_rgb, features=None):
        f = extract_c4_features(image_rgb) if features is None else features
        t = self.config
        # Conservative two-signal concurrence. This is a candidate detector, not ground truth.
        blur = (f["laplacian_variance"] < t["blur_laplacian_max"] and
                f["tenengrad"] < t["blur_tenengrad_max"])
        noise = (f["noise_mad"] > t["noise_mad_min"] and
                 f["wavelet_sigma"] > t["noise_wavelet_min"])
        return {"C4_1_blur": bool(blur), "C4_2_noise": bool(noise),
                "C4": bool(blur or noise), "features": f,
                "status": "experimental_unvalidated"}
