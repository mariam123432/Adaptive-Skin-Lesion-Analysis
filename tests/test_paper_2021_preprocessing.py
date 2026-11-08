"""Non-régression du prétraitement et adaptateur C1–C5 conservateur."""
from pathlib import Path
from threading import RLock
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
import torch

from src.inference.preprocessing import (
    SHADES, ADAPTIVE, AdaptivePreprocessor, prepare_image,
)
from src.inference.paper_2021 import Paper2021Segmenter
from src.paper_reproduction.preprocessing import shades_of_gray
from src.evaluation.paper_2021 import compare_summaries, selected_image_ids


def test_default_exactly_matches_original():
    rgb = np.random.default_rng(2).integers(0, 256, (70, 91, 3), dtype=np.uint8)
    expected = shades_of_gray(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA), 6)
    actual, metadata = prepare_image(rgb)
    np.testing.assert_array_equal(actual, expected)
    assert metadata['preprocessing_method'] == SHADES


def test_adaptive_never_calls_shades_and_uses_rgb_output():
    rgb = np.full((22, 29, 3), [10, 70, 120], np.uint8)
    adapter = Mock(return_value={'image': rgb + 5, 'steps_executed': ['C1'], 'adaptive_report': {}})
    with patch('src.inference.preprocessing.shades_of_gray', side_effect=AssertionError('double preprocessing')):
        actual, metadata = prepare_image(rgb, ADAPTIVE, adapter)
    np.testing.assert_array_equal(actual, cv2.resize(rgb + 5, (224, 224), interpolation=cv2.INTER_AREA))
    assert metadata['shades_of_gray_p'] is None
    adapter.assert_called_once()


def test_missing_or_invalid_adaptive_output():
    rgb = np.zeros((20, 30, 3), np.uint8)
    with pytest.raises(ValueError, match='adaptatif'):
        prepare_image(rgb, ADAPTIVE)
    adapter = Mock(return_value={'image': np.zeros((2, 3, 3), np.uint8), 'steps_executed': []})
    with pytest.raises(ValueError, match='dimensions'):
        prepare_image(rgb, ADAPTIVE, adapter)
    with pytest.raises(ValueError):
        prepare_image(rgb, 'unknown')


def test_real_c1_orchestrator_reused_and_other_corrections_disabled():
    adapter = AdaptivePreprocessor.__new__(AdaptivePreprocessor)
    adapter.hair = Mock()
    adapter.hair.predict_hair.return_value = {'hair_detected': True}
    adapter.detectors = {f'C{i}': Mock() for i in range(2, 6)}
    for key, detector in adapter.detectors.items():
        detector.predict.return_value = {key: True}
    adapter.model_files = {}
    rgb = np.full((40, 50, 3), 120, np.uint8)
    # Le vrai process_image choisit C1 et appelle DullRazor ; seul le résultat est contrôlé.
    cleaned = rgb - 10
    with patch('src.pipeline.dullrazor', return_value=(rgb, None, None, cleaned)) as dull:
        output = adapter(rgb)
    dull.assert_called_once()
    np.testing.assert_array_equal(output['image'], cleaned)
    assert output['steps_executed'] == ['C1: DullRazor adapté + Telea']
    assert output['adaptive_report']['C5']['correction_status'] == 'indisponible'
    for key in ['C2', 'C3', 'C4']:
        assert output['adaptive_report'][key]['correction_status'] == 'désactivée'


def test_same_segmenters_receive_selected_preprocessing():
    p = Paper2021Segmenter.__new__(Paper2021Segmenter)
    p.device, p._lock, p.model_files = torch.device('cpu'), RLock(), {}
    p.vgg, p.deep = Mock(), Mock()
    p.vgg.return_value = p.deep.return_value = torch.zeros((1, 1, 224, 224))
    rgb = np.full((30, 40, 3), 120, np.uint8)
    adapter = Mock(return_value={'image': rgb - 40, 'steps_executed': [], 'adaptive_report': {}})
    normal = p.predict(rgb)
    original_tensor = p.vgg.call_args.args[0].clone()
    adapted = p.predict(rgb, preprocessing=ADAPTIVE, adaptive=adapter)
    assert not torch.equal(original_tensor, p.vgg.call_args.args[0])
    assert normal.metadata['preprocessing_method'] == SHADES
    assert adapted.metadata['preprocessing_method'] == ADAPTIVE
    assert p.vgg.call_count == p.deep.call_count == 2


def test_comparison_deltas_and_selected_ids(tmp_path):
    a = {'task': 'classification', 'count': 2, 'accuracy': .5, 'balanced_accuracy': .4, 'macro_f1': .3}
    b = dict(a, accuracy=1., balanced_accuracy=.8, macro_f1=.7)
    rows = compare_summaries(a, b)
    assert rows[0]['adaptive_minus_shades'] == .5
    split = tmp_path / 'selected.csv'
    split.write_text('image_id,split\nISIC_0000001,test\n', encoding='utf-8')
    assert selected_image_ids(split) == ['ISIC_0000001']
    split.write_text('image_id\nISIC_0000001\nISIC_0000001\n', encoding='utf-8')
    with pytest.raises(ValueError, match='doublon'):
        selected_image_ids(split)


def test_comparison_same_identifiers_and_same_pipeline(tmp_path):
    from types import SimpleNamespace
    from src.evaluation.paper_2021 import evaluate_comparison
    image = tmp_path / 'ISIC_0000001.jpg'
    cv2.imwrite(str(image), np.zeros((20, 30, 3), np.uint8))
    pipeline = Mock()
    pipeline.predict.side_effect = lambda rgb, source, **kwargs: SimpleNamespace(
        metadata={'predicted_class': 'NV' if kwargs['preprocessing'] == SHADES else 'MEL',
                  'preprocessing_method': kwargs['preprocessing']})
    records = {'ISIC_0000001': {'image': image, 'label': 'NV'}}
    run = evaluate_comparison('classification', records, pipeline, Mock(), tmp_path / 'out', tmp_path / 'absent')
    assert pipeline.predict.call_count == 2
    assert [c.kwargs['preprocessing'] for c in pipeline.predict.call_args_list] == [SHADES, ADAPTIVE]
    assert run['comparison'][0]['adaptive_minus_shades'] == -1
    assert (run['directory'] / 'comparison.csv').is_file()
    assert run['runs'][SHADES]['metrics'][0]['image_id'] == run['runs'][ADAPTIVE]['metrics'][0]['image_id']


def test_buttons_invalidate_results_without_loading_models():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'demo/paper_2021_app.py').run(timeout=30)
    assert not app.exception
    assert app.session_state['paper_preprocessing_method'] == SHADES
    app.session_state['paper_result'] = {'obsolete': True}
    app.session_state['evaluation_result'] = {'obsolete': True}
    app.button(key='choose_adaptive_c1_c5').click().run(timeout=30)
    assert not app.exception
    assert app.session_state['paper_preprocessing_method'] == ADAPTIVE
    assert 'paper_result' not in app.session_state
    assert 'evaluation_result' not in app.session_state
    app.button(key='choose_shades_of_gray').click().run(timeout=30)
    assert not app.exception
    assert app.session_state['paper_preprocessing_method'] == SHADES
