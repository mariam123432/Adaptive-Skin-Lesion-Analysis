from pathlib import Path
import json
import joblib
from .hair_features import extract_hair_features, FEATURE_NAMES

MODEL_FILES = {"SVM RBF": "hair_svm_rbf.joblib", "Random Forest": "hair_random_forest.joblib"}

class HairPredictor:
    """Prédiction C1 à partir du modèle Kaggle, sans réentraînement."""
    def __init__(self, project_path, model_name=None):
        base = Path(project_path)
        config_path = base / "models" / "hair_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration manquante : {config_path}. Exporter le modèle depuis Kaggle.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if list(FEATURE_NAMES) != list(config.get("feature_names", [])):
            raise ValueError("Ordre des caractéristiques incompatible avec l'entraînement.")
        self.model_name = model_name or config.get("primary_model") or config.get("model") or "SVM RBF"
        if self.model_name not in MODEL_FILES:
            raise ValueError(f"Modèle inconnu : {self.model_name}")
        if "thresholds" in config:
            self.threshold = float(config["thresholds"][self.model_name])
        else:
            if self.model_name != (config.get("model") or "SVM RBF"):
                raise ValueError("Ancienne configuration : seuil indisponible pour ce modèle.")
            self.threshold = float(config["threshold"])
        model_path = base / "models" / MODEL_FILES[self.model_name]
        if not model_path.is_file():
            raise FileNotFoundError(f"Modèle manquant : {model_path}. Récupérer le .joblib sur Kaggle.")
        # joblib : ne charger que des fichiers provenant d'une source fiable.
        self.model = joblib.load(model_path)

    def predict_hair(self, image_rgb):
        features = extract_hair_features(image_rgb).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        positive_index = list(self.model.classes_).index(1)
        score = float(probabilities[positive_index])
        return {"classe": "C1_hair", "modele": self.model_name, "score": score,
                "seuil": self.threshold, "hair_detected": bool(score >= self.threshold)}
