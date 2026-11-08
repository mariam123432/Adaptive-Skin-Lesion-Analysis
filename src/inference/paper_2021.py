"""Inférence de l'adaptation expérimentale 2021 ; aucune modification des poids.

Les tableaux d'images sont RGB uint8 ; les masques binaires uint8 valent 0/1.
Les cartes de probabilités restent à la résolution de calcul 224 × 224.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import uuid4
import zipfile
import warnings

import cv2
import numpy as np
import torch

from src.paper_reproduction.settings import CLASSES
from src.paper_reproduction.preprocessing import shades_of_gray, normalize_tensor
from src.paper_reproduction.features import handcrafted200
from src.paper_reproduction.segmentation import make_segmenter
from src.paper_reproduction.diagnosis import HybridCNN
from src.inference.preprocessing import SHADES, prepare_image

ARTIFACTS = {
    'vgg': 'vgg19_unet_best.pth', 'deep': 'deeplabv3plus_best.pth',
    'resnet': 'hybrid_resnet50_best.pth', 'densenet': 'hybrid_densenet201_best.pth',
    'scaler': 'handcrafted_scaler.joblib', 'svm': 'hybrid_final_svm.joblib',
    'config': 'hybrid_final_svm_config.json',
}
NOTICE = 'Prototype expérimental de recherche — ne constitue pas un diagnostic médical.'
EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def validate_image(rgb):
    """Refuser les formats ambigus plutôt que transformer silencieusement les entrées."""
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
            rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
        raise ValueError('Image attendue : tableau RGB uint8 non vide (H, W, 3).')
    if rgb.shape[0] * rgb.shape[1] > 40_000_000:
        raise ValueError('Image trop grande : maximum 40 millions de pixels.')
    return rgb


def read_image(data: bytes, filename: str):
    """Décoder uniquement JPEG/PNG ; ne jamais stocker le fichier importé."""
    if Path(filename).suffix.lower() not in EXTENSIONS:
        raise ValueError('Extensions autorisées : .jpg, .jpeg, .png.')
    if not data or len(data) > 30 * 1024 * 1024:
        raise ValueError('Image vide ou supérieure à 30 Mo.')
    # Vérifier le vrai format et les dimensions avant la décompression OpenCV.
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in {'JPEG', 'PNG'}:
                raise ValueError('Le contenu doit être JPEG ou PNG.')
            if im.width * im.height > 40_000_000:
                raise ValueError('Image trop grande : maximum 40 millions de pixels.')
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError('Image invalide ou illisible.') from exc
    # Même lecture RGB que read_rgb du projet, y compris la gestion OpenCV des EXIF.
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Image invalide ou illisible.')
    return validate_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def class_label(index):
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(CLASSES):
        raise ValueError(f'Classe SVM incompatible : {index!r} ; entier 0–6 attendu.')
    return CLASSES[int(index)]


def fuse_probabilities(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or a.size == 0 or a.shape != b.shape:
        raise ValueError('Dimensions incompatibles des cartes de probabilités.')
    for p in (a, b):
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Probabilités invalides : valeurs finies dans [0, 1] attendues.')
    mean = (a.astype(np.float32) + b.astype(np.float32)) / 2.
    return mean, (mean >= .5).astype(np.uint8)


def check_artifacts(directory):
    paths = {key: Path(directory) / name for key, name in ARTIFACTS.items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Fichiers manquants :\n' + '\n'.join(missing))
    return paths


def load_checkpoint(model, path, device):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Checkpoint introuvable : {path}')
    try:
        # train_hybrid sérialise parfois sa métrique en numpy.float64.
        # Autoriser ces seuls types locaux, sans revenir à weights_only=False.
        from numpy.core.multiarray import scalar
        with torch.serialization.safe_globals([scalar, np.dtype, type(np.dtype('float64'))]):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f'Checkpoint incompatible : {path} : {exc}') from exc
    return model.to(device).eval()


def _matrix(values, shape, name):
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f'{name} incompatible : dimension attendue {shape}, reçue {values.shape}, valeurs finies requises.')
    return values.astype(np.float32)


def load_sklearn_artifact(path):
    """Refuser les objets dont la version sklearn ne garantit pas la compatibilité."""
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            return joblib.load(path)
    except InconsistentVersionWarning as exc:
        raise ValueError(
            f'Artefact sklearn incompatible : {path}. Version exportée '
            f'{exc.original_sklearn_version}, version installée {exc.current_sklearn_version}. '
            'Utiliser un environnement compatible avec la version exportée ; aucun poids à réentraîner.'
        ) from exc


def validate_svm(svm, config):
    """Le fichier réel contient StandardScaler → SVC ; conserver toute la chaîne."""
    from sklearn.pipeline import Pipeline
    if getattr(svm, 'n_features_in_', None) != 1024:
        raise ValueError('SVM incompatible : 1024 caractéristiques attendues.')
    if not np.array_equal(getattr(svm, 'classes_', None), np.arange(7)):
        raise ValueError('Classes SVM incompatibles : indices 0–6 attendus.')
    classifier = svm.steps[-1][1] if isinstance(svm, Pipeline) else svm
    params = classifier.get_params()
    if params.get('kernel') != config.get('kernel') or params.get('C') != config.get('C'):
        raise ValueError('Hyperparamètres SVM incompatibles avec la configuration.')


@dataclass
class Result:
    images: dict[str, np.ndarray]
    probabilities: dict[str, np.ndarray]
    manual_features: np.ndarray
    metadata: dict

    def json_bytes(self):
        return json.dumps(self.metadata, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')

    def files(self):
        """Exporter les masques en 0/255 ; conserver les cartes brutes dans un NPZ."""
        outputs = {'prediction.json': self.json_bytes()}
        for key, image in self.images.items():
            image = image * 255 if key.endswith('_mask') else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise ValueError(f'Échec encodage PNG : {key}')
            outputs[f'{key}.png'] = encoded.tobytes()
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **self.probabilities)
        outputs['probabilities.npz'] = buffer.getvalue()
        buffer = io.BytesIO()
        np.save(buffer, self.manual_features, allow_pickle=False)
        outputs['manual_features.npy'] = buffer.getvalue()
        return outputs

    def zip_bytes(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in self.files().items():
                archive.writestr(name, data)
        return buffer.getvalue()

    def save(self, directory):
        # mkdir exclusif : même deux sauvegardes simultanées ne peuvent s'écraser.
        target = Path(directory) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex)
        files = self.files()
        target.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            (target / name).write_bytes(data)
        return target


def _segment_probabilities(x, vgg, deep):
    """Calcul commun aux modes démonstration et évaluation, sans annotation GT."""
    probabilities = {}
    for name, model in [('vgg', vgg), ('deep', deep)]:
        model.eval()
        output = model(x)
        if tuple(output.shape) != (1, 1, 224, 224):
            raise ValueError(f'Sortie {name} incompatible : (1, 1, 224, 224) attendu, reçu {tuple(output.shape)}.')
        probabilities[name] = output.sigmoid()[0, 0].cpu().numpy()
    probabilities['fused'], _ = fuse_probabilities(probabilities['vgg'], probabilities['deep'])
    return probabilities


@dataclass
class SegmentationResult:
    images: dict
    probabilities: dict
    metadata: dict


class Paper2021Segmenter:
    """Évaluation Task 1 indépendante des hybrides, du scaler et du SVM."""
    def __init__(self, project, device='auto'):
        if device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('Device attendu : auto, cpu ou cuda.')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA demandé mais indisponible ; choisir CPU.')
        self.device = torch.device('cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu')
        folder = Path(project) / 'models/paper_2021'
        paths = {key: folder / ARTIFACTS[key] for key in ['vgg', 'deep']}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f'Checkpoint introuvable : {path}')
        self.vgg = load_checkpoint(make_segmenter('vgg19_unet', pretrained=False), paths['vgg'], self.device)
        self.deep = load_checkpoint(make_segmenter('deeplabv3plus', pretrained=False), paths['deep'], self.device)
        self.model_files = {key: str(path.resolve()) for key, path in paths.items()}
        self._lock = RLock()

    def predict(self, rgb, source='image', preprocessing=SHADES, adaptive=None):
        with self._lock, torch.inference_mode():
            rgb = validate_image(rgb).copy()
            normalized, preprocessing_info = prepare_image(rgb, preprocessing, adaptive)
            x = normalize_tensor(torch.from_numpy(normalized.transpose(2, 0, 1).copy()).float() / 255)[None].to(self.device)
            probabilities = _segment_probabilities(x, self.vgg, self.deep)
            h, w = rgb.shape[:2]
            images = {'original': rgb, 'preprocessed': normalized}
            for key in probabilities:
                images[f'{key}_mask'] = cv2.resize((probabilities[key] >= .5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            return SegmentationResult(images, probabilities, {
                'source': Path(source).name, 'device': str(self.device), 'model_files': self.model_files,
                'inference_size': [224, 224], 'shades_of_gray_p': 6, 'normalization': 'ImageNet mean/std',
                'segmentation_threshold': .5, 'fusion': 'mean(probability_vgg, probability_deep)',
                'mask_resize': 'threshold at 224x224 then nearest-neighbor to native size', 'note': NOTICE,
                **preprocessing_info})


class Paper2021Pipeline:
    """Charge une fois les quatre réseaux et les deux objets sklearn locaux."""

    def __init__(self, project, device='auto'):
        start = perf_counter()
        paths = check_artifacts(Path(project) / 'models/paper_2021')
        if device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('Device attendu : auto, cpu ou cuda.')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA demandé mais indisponible ; choisir CPU.')
        self.device = torch.device('cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu')
        self.config = json.loads(paths['config'].read_text(encoding='utf-8'))
        if (self.config.get('classes') != CLASSES or
                self.config.get('feature_source') != 'two_hybrid_CNN_embeddings_512x2' or
                self.config.get('manual_features') != 'adapted_handcrafted200'):
            raise ValueError('Configuration incompatible : classes, embeddings ou descripteur inconnus.')
        # Charger uniquement les artefacts de confiance livrés dans ce projet.
        self.scaler = load_sklearn_artifact(paths['scaler'])
        self.svm = load_sklearn_artifact(paths['svm'])
        if getattr(self.scaler, 'n_features_in_', None) != 200:
            raise ValueError('Scaler incompatible : 200 caractéristiques attendues.')
        validate_svm(self.svm, self.config)
        self.vgg = load_checkpoint(make_segmenter('vgg19_unet', pretrained=False), paths['vgg'], self.device)
        self.deep = load_checkpoint(make_segmenter('deeplabv3plus', pretrained=False), paths['deep'], self.device)
        self.resnet = load_checkpoint(HybridCNN('resnet50', pretrained=False), paths['resnet'], self.device)
        self.densenet = load_checkpoint(HybridCNN('densenet201', pretrained=False), paths['densenet'], self.device)
        self.model_files = {key: str(value.resolve()) for key, value in paths.items()}
        self.load_seconds = perf_counter() - start
        self._lock = RLock()

    def predict(self, rgb, source='image', preprocessing=SHADES, adaptive=None):
        # Streamlit partage cache_resource entre sessions : sérialiser l'accès aux réseaux.
        with getattr(self, '_lock', RLock()), torch.inference_mode():
            return self._predict(rgb, source, preprocessing, adaptive)

    def _predict(self, rgb, source, preprocessing=SHADES, adaptive=None):
        timings = {}
        def tick():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            return perf_counter()
        start = tick()
        rgb = validate_image(rgb).copy()
        normalized, preprocessing_info = prepare_image(rgb, preprocessing, adaptive)
        x = normalize_tensor(torch.from_numpy(normalized.transpose(2, 0, 1).copy()).float() / 255)[None].to(self.device)
        timings['preprocessing'] = tick() - start
        t = tick()
        probabilities = _segment_probabilities(x, self.vgg, self.deep)
        mask = (probabilities['fused'] >= .5).astype(np.uint8)
        timings['segmentation'] = tick() - t
        t = tick()
        manual = _matrix(handcrafted200(normalized, mask), (200,), 'Descripteur 200')
        scaled = _matrix(self.scaler.transform(manual[None]), (1, 200), 'Scaler 200')
        f = torch.from_numpy(scaled).to(self.device)
        timings['manual_features'] = tick() - t
        t = tick()
        embeddings = []
        for name, model in [('ResNet50', self.resnet), ('DenseNet201', self.densenet)]:
            model.eval()
            embeddings.append(_matrix(model.embedding(x, f).cpu().numpy(), (1, 512), f'Embedding {name} 512'))
        combined = _matrix(np.concatenate(embeddings, axis=1), (1, 1024), 'Concaténation 1024')
        prediction = np.asarray(self.svm.predict(combined))
        if prediction.shape != (1,):
            raise ValueError('Le SVM doit retourner une seule classe.')
        label = class_label(prediction[0])
        timings['classification'] = tick() - t
        h, w = rgb.shape[:2]
        images = {'original': rgb, 'preprocessed': normalized}
        # Le seuil est appliqué à 224² puis le masque est agrandi sans interpolation des labels.
        for key in ['vgg', 'deep', 'fused']:
            images[f'{key}_mask'] = cv2.resize((probabilities[key] >= .5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = rgb.copy()
        selected = images['fused_mask'] > 0
        overlay[selected] = (.7 * rgb[selected] + .3 * np.array([0, 255, 0])).astype(np.uint8)
        images['overlay'] = overlay
        timings['total'] = tick() - start
        metadata = {
            'source': Path(source).name, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'predicted_class': label, 'class_index': int(prediction[0]), 'classes': CLASSES,
            'device': str(self.device), 'input_shape': list(rgb.shape), 'inference_size': [224, 224],
            'shades_of_gray_p': 6, 'normalization': 'ImageNet mean/std',
            'segmentation_threshold': .5, 'fusion': 'mean(probability_vgg, probability_deep)',
            'mask_resize': 'threshold at 224x224 then nearest-neighbor to native size',
            'manual_feature_dim': 200, 'embedding_dims': [512, 512], 'svm_feature_dim': 1024,
            'embedding_order': ['ResNet50', 'DenseNet201'], 'model_files': self.model_files,
            'load_seconds': self.load_seconds, 'timings_seconds': timings, 'configuration': self.config,
            'empty_mask': not bool(mask.any()), 'note': NOTICE,
            'implementation': 'Adaptation expérimentale ; handcrafted200 non identique au descripteur original du papier.',
            **preprocessing_info,
        }
        return Result(images, probabilities, manual, metadata)
