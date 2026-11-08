"""C3: décision multi-label à partir des mesures et d'un JSON calibrable.
Aucune étiquette médicale ni masque de lésion n'est utilisée en inférence.
"""
from pathlib import Path
import json
from .c3_features import extract_c3_features

DEFAULT_THRESHOLDS={
    "max_low_contrast_rms":0.075,
    "max_low_contrast_range":0.22,
    "min_uneven_illumination_range":0.20,
    "min_uneven_quadrant_range":0.18,
    "min_underexposed_mean":0.27,
    "max_overexposed_mean":0.78,
    "max_dark_ratio":0.12,
    "max_bright_ratio":0.10,
}

def classify_features(f, thresholds):
    t=thresholds
    low=(f["contrast_rms"] < t["max_low_contrast_rms"] and
         f["contrast_p95_p5"] < t["max_low_contrast_range"])
    uneven=(f["illumination_range"] > t["min_uneven_illumination_range"] and
            f["quadrant_mean_range"] > t["min_uneven_quadrant_range"])
    under=(f["mean_luma"] < t["min_underexposed_mean"] or
           f["dark_ratio"] > t["max_dark_ratio"])
    over=(f["mean_luma"] > t["max_overexposed_mean"] or
          f["bright_ratio"] > t["max_bright_ratio"])
    return {"C3_1_low_contrast":bool(low), "C3_2_uneven_illumination":bool(uneven),
            "C3_3_underexposed":bool(under),"C3_3_overexposed":bool(over),
            "C3":bool(low or uneven or under or over)}

class C3Detector:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else None
        if self.config_path:
            with self.config_path.open(encoding="utf-8") as f: config=json.load(f)
            self.thresholds=config["thresholds"]
        else:
            self.thresholds=DEFAULT_THRESHOLDS.copy()
    def predict(self, image_rgb):
        f=extract_c3_features(image_rgb)
        return {"features":f, **classify_features(f,self.thresholds)}
