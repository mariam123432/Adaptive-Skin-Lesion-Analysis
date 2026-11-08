"""Fusion multi-étiquettes C1–C5. Les classes non évaluées restent None."""

def classify_problems(image_rgb, hair_predictor=None, other_detectors=None):
    labels = {f"C{i}": None for i in range(1, 6)}
    details = {}
    if hair_predictor is not None:
        result = hair_predictor.predict_hair(image_rgb)
        labels["C1"] = result["hair_detected"]
        details["hair"] = result
    for class_name, detector in (other_detectors or {}).items():
        if class_name not in labels or class_name == "C1":
            raise ValueError(f"Classe invalide ou déjà traitée : {class_name}")
        outcome = detector(image_rgb)
        if not isinstance(outcome, bool):
            raise TypeError(f"Le détecteur de {class_name} doit retourner un booléen")
        labels[class_name] = outcome
    return {"classes": labels, "details": details}
