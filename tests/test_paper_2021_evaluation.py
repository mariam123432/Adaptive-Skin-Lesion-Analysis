"""Tests sur tableaux et fichiers temporaires ; aucune annotation inventée sur disque projet."""
import csv
import json
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.evaluation.paper_2021 import (
    segmentation_metrics, pair_segmentation, read_task3_labels, audit_partition,
    load_ground_truth, error_overlay, classification_summary, evaluate_batch,
)


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_dice_iou_counts_and_empty():
    a = np.array([[1, 1], [0, 0]], np.uint8)
    b = np.array([[1, 0], [1, 0]], np.uint8)
    m = segmentation_metrics(a, b)
    assert m == {'dice': .5, 'iou': 1 / 3, 'tp': 1, 'fp': 1, 'fn': 1, 'tn': 1}
    assert segmentation_metrics(a, a)['dice'] == 1
    assert segmentation_metrics(a, 1 - a)['iou'] == 0
    assert segmentation_metrics(a * 0, a * 0)['dice'] == 1
    with pytest.raises(ValueError):
        segmentation_metrics(a, b[:1])
    with pytest.raises(ValueError):
        segmentation_metrics(a * 2, b)


def test_pair_exact_id_and_duplicates(tmp_path):
    images, masks = tmp_path / 'images', tmp_path / 'masks'
    images.mkdir(); masks.mkdir()
    (images / 'ISIC_0024306.jpg').touch()
    (images / 'ISIC_0024307.jpg').touch()
    (masks / 'ISIC_0024306_segmentation.png').touch()
    pairs, issues = pair_segmentation(images, masks)
    assert list(pairs) == ['ISIC_0024306']
    assert any('ISIC_0024307' in issue for issue in issues)
    (images / 'ISIC_0024306.JPG').touch()
    # Windows insensible à la casse : utiliser une autre extension pour le doublon.
    (images / 'ISIC_0024306.jpeg').touch()
    with pytest.raises(ValueError, match='doublon'):
        pair_segmentation(images, masks)


def test_task3_strict_one_hot(tmp_path):
    path = tmp_path / 'gt.csv'
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    row = dict(image='ISIC_0024306', **{c: int(c == 'NV') for c in classes})
    write_csv(path, ['image'] + classes, [row])
    assert read_task3_labels(path) == {'ISIC_0024306': 'NV'}
    row['MEL'] = 1
    write_csv(path, ['image'] + classes, [row])
    with pytest.raises(ValueError, match='one-hot'):
        read_task3_labels(path)
    row['MEL'] = 0
    write_csv(path, ['image'] + classes, [row, row])
    with pytest.raises(ValueError, match='doublon'):
        read_task3_labels(path)


def test_partition_and_lesion_leakage(tmp_path):
    split = tmp_path / 'splits.csv'
    write_csv(split, ['image_id', 'split'], [
        {'image_id': 'ISIC_0000001', 'split': 'train'},
        {'image_id': 'ISIC_0000002', 'split': 'test'}])
    metadata = tmp_path / 'metadata.csv'
    write_csv(metadata, ['image_id', 'lesion_id'], [
        {'image_id': 'ISIC_0000001', 'lesion_id': 'lesion-a'},
        {'image_id': 'ISIC_0000002', 'lesion_id': 'lesion-a'}])
    audit = audit_partition(['ISIC_0000002'], split, metadata)
    assert audit['all_test'] and not audit['generalization_eligible']
    assert audit['lesion_leaks'] == ['ISIC_0000002']
    assert not audit_partition(['ISIC_0000001'], split)['all_test']
    assert 'Partition non vérifiée' in audit_partition(['ISIC_0000002'], tmp_path / 'absent')['message']


def test_ground_truth_resize_and_overlay(tmp_path):
    path = tmp_path / 'mask.png'
    cv2.imwrite(str(path), np.array([[0, 255], [255, 0]], np.uint8))
    gt, resized = load_ground_truth(path, (4, 4))
    assert resized and gt.shape == (4, 4) and set(np.unique(gt)) == {0, 1}
    image = np.zeros((2, 2, 3), np.uint8)
    overlay = error_overlay(image, np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]))
    assert overlay[0, 0, 1] > 0   # TP vert
    assert overlay[0, 1, 0] > 0   # FP rouge
    assert overlay[1, 0, 2] > 0   # FN bleu


def test_classification_report():
    summary = classification_summary(['MEL', 'NV'], ['MEL', 'MEL'])
    assert summary['accuracy'] == .5
    assert summary['balanced_accuracy'] == .5
    assert np.asarray(summary['confusion_matrix']).shape == (7, 7)
    assert summary['report']['NV']['recall'] == 0


def test_batch_never_passes_gt_and_unique_export(tmp_path):
    image = tmp_path / 'ISIC_0000002.jpg'
    mask = tmp_path / 'ISIC_0000002_segmentation.png'
    cv2.imwrite(str(image), np.zeros((4, 5, 3), np.uint8))
    cv2.imwrite(str(mask), np.zeros((4, 5), np.uint8))
    pipeline = Mock()
    pipeline.predict.return_value = SimpleNamespace(
        images={f'{key}_mask': np.zeros((4, 5), np.uint8) for key in ['vgg', 'deep', 'fused']},
        metadata={'device': 'cpu'})
    records = {'ISIC_0000002': {'image': image, 'mask': mask}}
    run = evaluate_batch('segmentation', records, pipeline, tmp_path / 'out', tmp_path / 'missing.csv')
    assert pipeline.predict.call_args.args[0].shape == (4, 5, 3)
    assert len(pipeline.predict.call_args.args) == 2
    assert run['summary']['means']['fused']['dice'] == 1
    assert (run['directory'] / 'metrics.csv').is_file()
    assert json.loads((run['directory'] / 'summary.json').read_text(encoding='utf-8'))['count'] == 1
    run2 = evaluate_batch('segmentation', records, pipeline, tmp_path / 'out', tmp_path / 'missing.csv')
    assert run2['directory'] != run['directory']


def test_classification_batch_and_failed_manifest(tmp_path):
    image = tmp_path / 'ISIC_0000002.jpg'
    cv2.imwrite(str(image), np.zeros((4, 5, 3), np.uint8))
    records = {'ISIC_0000002': {'image': image, 'label': 'NV'}}
    pipeline = Mock()
    pipeline.predict.return_value = SimpleNamespace(metadata={'predicted_class': 'MEL'})
    run = evaluate_batch('classification', records, pipeline, tmp_path / 'out', tmp_path / 'missing.csv')
    assert run['metrics'][0]['correct'] is False
    assert run['summary']['accuracy'] == 0
    assert (run['directory'] / 'classification_report.csv').is_file()
    assert (run['directory'] / 'confusion_matrix.csv').is_file()
    pipeline.predict.side_effect = ValueError('checkpoint incompatible')
    with pytest.raises(RuntimeError, match='interrompue'):
        evaluate_batch('classification', records, pipeline, tmp_path / 'failed', tmp_path / 'missing.csv')
    manifest = next((tmp_path / 'failed').rglob('manifest.json'))
    assert json.loads(manifest.read_text(encoding='utf-8'))['status'] == 'failed'
    assert not (manifest.parent / 'summary.json').exists()


def test_segmentation_only_reuses_inference_without_svm():
    import torch
    from threading import RLock
    from src.inference.paper_2021 import Paper2021Segmenter
    p = Paper2021Segmenter.__new__(Paper2021Segmenter)
    p._lock, p.device, p.model_files = RLock(), torch.device('cpu'), {}
    p.vgg, p.deep = Mock(), Mock()
    p.vgg.return_value = torch.full((1, 1, 224, 224), -2.)
    p.deep.return_value = torch.full((1, 1, 224, 224), 3.)
    result = p.predict(np.zeros((4, 6, 3), np.uint8))
    assert result.images['fused_mask'].shape == (4, 6)
    np.testing.assert_allclose(result.probabilities['fused'],
                               (torch.tensor(-2.).sigmoid() + torch.tensor(3.).sigmoid()).item() / 2)
    assert not hasattr(p, 'svm')


def test_invalid_mask_and_aspect_ratio(tmp_path):
    path = tmp_path / 'mask.png'
    cv2.imwrite(str(path), np.zeros((2, 4), np.uint8))
    with pytest.raises(ValueError, match='Rapport de forme'):
        load_ground_truth(path, (4, 4))
    cv2.imwrite(str(path), np.full((2, 2), 127, np.uint8))
    with pytest.raises(ValueError, match='binaire'):
        load_ground_truth(path, (2, 2))


@pytest.mark.parametrize('orientation', [3, 6])
def test_rotated_exif_not_silently_compared(tmp_path, orientation):
    from PIL import Image
    image = tmp_path / 'ISIC_0000002.jpg'
    mask = tmp_path / 'ISIC_0000002_segmentation.png'
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(np.zeros((4, 5, 3), np.uint8)).save(image, exif=exif)
    cv2.imwrite(str(mask), np.zeros((4, 5), np.uint8))
    pipeline = Mock()
    with pytest.raises(RuntimeError, match='EXIF'):
        evaluate_batch('segmentation', {'ISIC_0000002': {'image': image, 'mask': mask}},
                       pipeline, tmp_path / 'out', tmp_path / 'missing.csv')
    pipeline.predict.assert_not_called()
