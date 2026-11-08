"""Contrats d'inférence : doubles légers, aucun téléchargement ni gros modèle."""
import io
import json
import os
import zipfile
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

from src.inference.paper_2021 import (
    Paper2021Pipeline, Result, class_label, fuse_probabilities, read_image,
    validate_image, load_checkpoint, check_artifacts, validate_svm, load_sklearn_artifact,
)


@pytest.mark.parametrize('image', [None, np.zeros((4, 4)), np.zeros((4, 4, 4)),
                                  np.zeros((0, 4, 3), dtype=np.uint8),
                                  np.zeros((4, 4, 3), dtype=np.float32)])
def test_invalid_arrays(image):
    with pytest.raises(ValueError):
        validate_image(image)


def test_decode_and_extensions(tmp_path):
    rgb = np.full((12, 16, 3), [20, 60, 90], dtype=np.uint8)
    data = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
    assert np.array_equal(read_image(data, 'test.PNG'), rgb)
    for data, name in [(b'broken', 'a.png'), (data, 'a.txt'), (b'', 'a.jpg')]:
        with pytest.raises(ValueError):
            read_image(data, name)


def test_probability_mean_and_mask():
    a = np.array([[.1, .4], [.8, 1]], dtype=np.float32)
    b = np.array([[.3, .6], [.2, 0]], dtype=np.float32)
    mean, mask = fuse_probabilities(a, b)
    np.testing.assert_allclose(mean, (a + b) / 2)
    assert mask.dtype == np.uint8 and mask.shape == a.shape
    np.testing.assert_array_equal(mask, [[0, 1], [1, 1]])


@pytest.mark.parametrize('b', [np.zeros((3, 2)), np.full((2, 2), np.nan), np.full((2, 2), 2)])
def test_bad_probabilities(b):
    with pytest.raises(ValueError):
        fuse_probabilities(np.zeros((2, 2)), b)


def test_labels():
    assert [class_label(i) for i in range(7)] == ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    for label in [-1, 7, .5, 'NV', True]:
        with pytest.raises(ValueError):
            class_label(label)


def test_missing_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError, match='vgg19_unet_best.pth'):
        check_artifacts(tmp_path)


def test_checkpoint_strict(tmp_path):
    path = tmp_path / 'weights.pth'
    torch.save({'state_dict': torch.nn.Linear(2, 1).state_dict(), 'val_balanced_acc': np.float64(.5)}, path)
    load_checkpoint(torch.nn.Linear(2, 1), path, torch.device('cpu'))
    with pytest.raises(ValueError, match='incompatible'):
        load_checkpoint(torch.nn.Linear(3, 1), path, torch.device('cpu'))


def test_real_sklearn_pipeline_format():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    svm = make_pipeline(StandardScaler(), SVC(C=.1))
    # Représenter les métadonnées ajustées sans entraîner les réseaux.
    svm[0].n_features_in_ = 1024
    svm[-1].classes_ = np.arange(7)
    validate_svm(svm, {'kernel': 'rbf', 'C': .1})
    with pytest.raises(ValueError, match='Hyperparamètres'):
        validate_svm(svm, {'kernel': 'linear', 'C': .1})
    svm[0].n_features_in_ = 1023
    with pytest.raises(ValueError, match='1024'):
        validate_svm(svm, {'kernel': 'rbf', 'C': .1})


def test_sklearn_version_mismatch(monkeypatch):
    import joblib
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning
    def mismatch(_):
        warnings.warn(InconsistentVersionWarning(estimator_name='SVC',
                      current_sklearn_version='1.9.1', original_sklearn_version='1.6.1'))
    monkeypatch.setattr(joblib, 'load', mismatch)
    with pytest.raises(ValueError, match='1.6.1'):
        load_sklearn_artifact('local.joblib')


def fake_pipeline():
    p = Paper2021Pipeline.__new__(Paper2021Pipeline)
    p.device = torch.device('cpu')
    p.config = {'classes': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']}
    p.model_files = {}
    p.load_seconds = 0.
    def segment(logit):
        model = Mock()
        model.return_value = torch.full((1, 1, 224, 224), logit)
        return model
    p.vgg, p.deep = segment(-1.), segment(2.)
    p.resnet, p.densenet = Mock(), Mock()
    p.resnet.embedding.return_value = torch.ones((1, 512))
    p.densenet.embedding.return_value = torch.full((1, 512), 2.)
    p.scaler = Mock(n_features_in_=200)
    p.scaler.transform.side_effect = lambda x: x
    p.svm = Mock(n_features_in_=1024)
    p.svm.predict.return_value = np.array([1])
    return p


def test_inference_order_shapes_and_serialization(tmp_path):
    p = fake_pipeline()
    result = p.predict(np.full((20, 30, 3), 120, dtype=np.uint8), 'test.png')
    assert result.metadata['predicted_class'] == 'NV'
    for key in ['vgg_mask', 'deep_mask', 'fused_mask']:
        assert result.images[key].shape == (20, 30)
        assert result.images[key].dtype == np.uint8
        assert set(np.unique(result.images[key])) <= {0, 1}
    combined = p.svm.predict.call_args.args[0]
    np.testing.assert_array_equal(combined[:, :512], 1)
    np.testing.assert_array_equal(combined[:, 512:], 2)
    assert result.manual_features.shape == (200,)
    assert json.loads(result.json_bytes())['predicted_class'] == 'NV'
    with zipfile.ZipFile(io.BytesIO(result.zip_bytes())) as archive:
        assert len(archive.namelist()) == 9
        assert 'prediction.json' in archive.namelist()
    first, second = result.save(tmp_path), result.save(tmp_path)
    assert first != second
    assert (first / 'prediction.json').is_file()


def test_bad_embedding_and_scaler():
    p = fake_pipeline()
    p.resnet.embedding.return_value = torch.ones((1, 511))
    with pytest.raises(ValueError, match='512'):
        p.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    p = fake_pipeline()
    p.scaler.transform.return_value = np.zeros((1, 199))
    p.scaler.transform.side_effect = None
    with pytest.raises(ValueError, match='200'):
        p.predict(np.zeros((10, 10, 3), dtype=np.uint8))


def test_full_pipeline_selected_image_reaches_manual_and_hybrids():
    from unittest.mock import patch
    from src.inference.preprocessing import SHADES, ADAPTIVE
    from src.paper_reproduction.features import handcrafted200
    p = fake_pipeline()
    rgb = np.full((24, 36, 3), [40, 90, 180], np.uint8)
    first = p.predict(rgb)
    explicit = p.predict(rgb, preprocessing=SHADES)
    for key in first.images:
        np.testing.assert_array_equal(first.images[key], explicit.images[key])
    np.testing.assert_array_equal(first.manual_features, explicit.manual_features)
    networks = (p.vgg, p.deep, p.resnet, p.densenet, p.svm)
    changed = np.full_like(rgb, [90, 80, 40])
    adapter = Mock(return_value={'image': changed, 'steps_executed': ['C1'], 'adaptive_report': {}})
    with patch('src.inference.preprocessing.shades_of_gray', side_effect=AssertionError('double preprocessing')):
        with patch('src.inference.paper_2021.handcrafted200', wraps=handcrafted200) as manual:
            adapted = p.predict(rgb, preprocessing=ADAPTIVE, adaptive=adapter)
    expected = cv2.resize(changed, (224, 224), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(manual.call_args.args[0], expected)
    np.testing.assert_array_equal(adapted.images['preprocessed'], expected)
    image_tensor = p.vgg.call_args.args[0]
    assert torch.equal(image_tensor, p.deep.call_args.args[0])
    assert torch.equal(image_tensor, p.resnet.embedding.call_args.args[0])
    assert torch.equal(image_tensor, p.densenet.embedding.call_args.args[0])
    assert networks == (p.vgg, p.deep, p.resnet, p.densenet, p.svm)


@pytest.mark.skipif(os.environ.get('PAPER2021_INTEGRATION') != '1', reason='Intégration facultative')
def test_real_local_models():
    root = Path(__file__).resolve().parents[1]
    images = [p for p in (root / 'test_images').glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not images:
        pytest.skip('Aucune image locale')
    try:
        check_artifacts(root / 'models/paper_2021')
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    image = images[0]
    result = Paper2021Pipeline(root, 'cpu').predict(read_image(image.read_bytes(), image.name), image.name)
    assert result.metadata['predicted_class'] in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
