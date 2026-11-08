"""Choix exclusif de prétraitement ; les modules scientifiques d'origine restent intacts."""
from pathlib import Path
from time import perf_counter
import warnings

import cv2
import numpy as np

from src.paper_reproduction.preprocessing import shades_of_gray

SHADES = 'shades_of_gray'
ADAPTIVE = 'adaptive_c1_c5'
METHOD_LABELS = {SHADES: 'Shades of Gray (papier 2021)', ADAPTIVE: 'Mon prétraitement adaptatif C1–C5'}


def _serializable(value):
    """Les masques des détecteurs sont décrits, sans gonfler le JSON de millions de valeurs."""
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'nonzero': int(np.count_nonzero(value))}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return value


class AdaptivePreprocessor:
    """Politique approuvée : C1 automatique ; C2–C5 détectés, corrections non activées.

    Ces détecteurs concernent les artefacts, jamais les sept classes diagnostiques.
    Aucun changement n'est fait à process_image ni aux détecteurs historiques.
    """
    def __init__(self, project):
        from sklearn.exceptions import InconsistentVersionWarning
        from src.classification.hair_predictor import HairPredictor
        from src.classification.c2_gel_classifier import C2Detector
        from src.classification.c3_detector import C3Detector
        from src.classification.c4_detector import C4Detector
        from src.classification.c5_classifier import C5Detector
        project = Path(project)
        names = ['hair_config.json', 'c2_config.json', 'c2_reflections_config.json',
                 'c3_config.json', 'c4_config.json', 'c5_config.json', 'hair_svm_rbf.joblib',
                 'c2_gel_bubble.joblib', 'c2_gel_border.joblib', 'c5_ruler.joblib', 'c5_dark_corner.joblib']
        self.model_files = {name: str((project / 'models' / name).resolve()) for name in names}
        for path in self.model_files.values():
            if not Path(path).is_file():
                raise FileNotFoundError(f'Prétraitement adaptatif indisponible : fichier manquant {path}')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', InconsistentVersionWarning)
                self.hair = HairPredictor(project, 'SVM RBF')
                self.detectors = {'C2': C2Detector(project), 'C3': C3Detector(project / 'models/c3_config.json'),
                                  'C4': C4Detector(project), 'C5': C5Detector(project)}
        except InconsistentVersionWarning as exc:
            raise ValueError(f'Prétraitement adaptatif incompatible : modèles sklearn {exc.original_sklearn_version}, '
                             f'version installée {exc.current_sklearn_version}. Utiliser un environnement compatible ; '
                             'aucune prédiction adaptative de substitution.') from exc

    def __call__(self, rgb):
        from src.inference.paper_2021 import validate_image
        from src.pipeline import process_image
        rgb = validate_image(rgb)
        if min(rgb.shape[:2]) < 16:
            raise ValueError('Prétraitement adaptatif : minimum 16 pixels par dimension pour C3.')
        # L'orchestrateur d'origine décide de l'application de DullRazor, sur l'image native.
        result = process_image(rgb.copy(), self.hair)
        report = {'C1': {'detected': result['classes']['C1'], 'details': result['details'],
                         'correction_status': 'appliquée' if result['applied'] else 'non nécessaire'}}
        # Évaluer les autres artefacts sur l'original, sans les confondre avec des corrections.
        for name, detector in self.detectors.items():
            prediction = detector.predict(rgb.copy())
            report[name] = {'detected': bool(prediction[name]), 'details': _serializable(prediction),
                            'correction_status': 'indisponible' if name == 'C5' else 'désactivée'}
        return {'image': result['cleaned_image'], 'steps_executed': result['applied'],
                'adaptive_report': report, 'adaptive_model_files': self.model_files,
                'adaptive_policy': 'C1 automatique ; corrections C2/C3/C4 désactivées ; C5 sans correction disponible.'}


def prepare_image(rgb, method=SHADES, adaptive=None):
    """Retourner exactement l'image RGB uint8 224² qui sera normalisée pour les réseaux."""
    from src.inference.paper_2021 import validate_image
    start = perf_counter()
    validate_image(rgb)
    if method not in METHOD_LABELS:
        raise ValueError(f'Méthode de prétraitement inconnue : {method}')
    if method == SHADES:
        # Ordre et interpolation inchangés par rapport à l'entraînement et à l'ancienne démo.
        image = shades_of_gray(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA), p=6)
        details = {'steps_executed': ['resize_224_INTER_AREA', 'Shades of Gray p=6']}
    else:
        if adaptive is None:
            raise ValueError('Prétraitement adaptatif indisponible : adaptateur non chargé.')
        details = dict(adaptive(rgb.copy()))
        image = validate_image(details.pop('image'))
        if image.shape != rgb.shape:
            raise ValueError('Prétraitement adaptatif incompatible : les dimensions natives doivent être conservées.')
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        details['steps_executed'] = list(details.get('steps_executed', [])) + ['resize_224_INTER_AREA']
    details.update(preprocessing_method=method, preprocessing_label=METHOD_LABELS[method],
                   shades_of_gray_p=6 if method == SHADES else None,
                   preprocessing_seconds=perf_counter() - start)
    return image, details
