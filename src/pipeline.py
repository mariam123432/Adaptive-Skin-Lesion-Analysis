"""Orchestrateur indépendant du notebook Kaggle."""
import cv2
from .classification.quality import classify_problems
from .preprocessing.hair_removal import dullrazor


def process_image(image_rgb, hair_predictor=None, other_detectors=None):
    """Classification puis correction C1; autres classes signalées, jamais simulées."""
    report = classify_problems(image_rgb, hair_predictor, other_detectors)
    output = image_rgb.copy()
    applied, pending = [], []
    if report["classes"]["C1"] is True:
        _, _, mask, cleaned = dullrazor(image_rgb)
        # La version Kaggle redimensionne à largeur 768 : garder une taille compatible
        # avec l'image source. Cette interpolation ne restaure pas les détails perdus.
        h, w = image_rgb.shape[:2]
        output = cv2.resize(cleaned, (w, h)) if cleaned.shape[:2] != (h, w) else cleaned
        applied.append("C1: DullRazor adapté + Telea")
    for label in ["C2", "C3", "C4", "C5"]:
        if report["classes"][label] is True:
            pending.append(f"{label}: correction non implémentée")
    return {"classes": report["classes"], "details": report["details"],
            "cleaned_image": output, "applied": applied, "pending": pending}
