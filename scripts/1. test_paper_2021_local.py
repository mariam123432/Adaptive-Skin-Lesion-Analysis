"""Test local d'inférence du pipeline Kaggle Ali & Ragb (2021), adaptation expérimentale.

Lancer à la racine du projet :
  python -m scripts.test_paper_2021_local --image test_images/ma_lesion.jpg

Sorties dans results/paper_2021/local_tests/<nom_image>/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from src.paper_reproduction.settings import CLASSES
from src.paper_reproduction.data import read_rgb
from src.paper_reproduction.preprocessing import shades_of_gray, normalize_tensor
from src.paper_reproduction.segmentation import make_segmenter, ensemble_mask
from src.paper_reproduction.features import handcrafted200
from src.paper_reproduction.diagnosis import HybridCNN


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")
    # Les checkpoints de ce notebook stockent {'state_dict': ..., 'epoch': ...}.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def predict(image_path: Path, project: Path, device: torch.device) -> dict:
    models_dir = project / 'models' / 'paper_2021'
    paths = {
        'vgg': models_dir / 'vgg19_unet_best.pth',
        'deep': models_dir / 'deeplabv3plus_best.pth',
        'resnet': models_dir / 'hybrid_resnet50_best.pth',
        'densenet': models_dir / 'hybrid_densenet201_best.pth',
        'scaler': models_dir / 'handcrafted_scaler.joblib',
        'svm': models_dir / 'hybrid_final_svm.joblib',
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Fichiers manquants :\n' + '\n'.join(missing))
    rgb = read_rgb(image_path)
    h, w = rgb.shape[:2]

    # Important : pretrained=False pour éviter un re-téléchargement ImageNet.
    vgg = load_checkpoint(make_segmenter('vgg19_unet', pretrained=False), paths['vgg'], device)
    deep = load_checkpoint(make_segmenter('deeplabv3plus', pretrained=False), paths['deep'], device)
    res = load_checkpoint(HybridCNN('resnet50', pretrained=False), paths['resnet'], device)
    den = load_checkpoint(HybridCNN('densenet201', pretrained=False), paths['densenet'], device)
    scaler = joblib.load(paths['scaler'])
    svm = joblib.load(paths['svm'])

    # Preprocessing identique aux entraînements Kaggle : RGB, 224², Shades of Gray p=6.
    size = 224
    normalized_rgb = shades_of_gray(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA), p=6)
    x = torch.from_numpy(normalized_rgb.transpose(2, 0, 1).copy()).float() / 255.0
    x = normalize_tensor(x).unsqueeze(0).to(device)

    with torch.inference_mode():
        # La fusion est réalisée sur les probabilités des deux segmentateurs.
        vgg_prob = torch.sigmoid(vgg(x))[0, 0].cpu().numpy()
        deep_prob = torch.sigmoid(deep(x))[0, 0].cpu().numpy()
        fused_prob = (vgg_prob + deep_prob) / 2.0
        mask_224 = (fused_prob >= 0.5).astype(np.uint8)

        # Le descripteur doit correspondre exactement aux 200 dimensions du notebook Kaggle.
        raw_features = handcrafted200(normalized_rgb, mask_224)
        features = scaler.transform(raw_features[None, :]).astype(np.float32)
        f = torch.from_numpy(features).to(device)
        e1 = res.embedding(x, f).cpu().numpy()
        e2 = den.embedding(x, f).cpu().numpy()
        combined = np.concatenate([e1, e2], axis=1)
        idx = int(svm.predict(combined)[0])

    mask_native = cv2.resize(mask_224, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = rgb.copy()
    alpha = 0.3
    tint = np.zeros_like(rgb)
    tint[:, :, 1] = 255
    overlay[mask_native > 0] = (
        (1 - alpha) * overlay[mask_native > 0] + alpha * tint[mask_native > 0]
    ).astype(np.uint8)

    dst = project / 'results' / 'paper_2021' / 'local_tests' / image_path.stem
    dst.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst / '01_original.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(dst / '02_shades_of_gray.png'), cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(dst / '03_vgg19_unet_mask.png'), (vgg_prob >= .5).astype(np.uint8) * 255)
    cv2.imwrite(str(dst / '04_deeplabv3plus_mask.png'), (deep_prob >= .5).astype(np.uint8) * 255)
    cv2.imwrite(str(dst / '05_ensemble_mask.png'), mask_native * 255)
    cv2.imwrite(str(dst / '06_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    record = {
        'image': str(image_path),
        'predicted_class': CLASSES[idx],
        'class_index': idx,
        'segmentation_threshold': 0.5,
        'device': str(device),
        'output_dir': str(dst),
        'note': 'Prediction expérimentale; aucun usage diagnostique clinique.',
    }
    (dst / 'prediction.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, type=Path, help='Image dermoscopique locale .jpg/.png')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.image.is_absolute():
        args.image = root / args.image
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA demandé, mais non disponible : utiliser --device cpu')
    device = torch.device('cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    record = predict(args.image, root, device)
    print(json.dumps(record, ensure_ascii=False, indent=2))