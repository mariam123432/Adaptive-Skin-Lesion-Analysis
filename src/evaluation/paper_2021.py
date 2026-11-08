"""Métriques et évaluations ISIC : les annotations ne sont jamais transmises aux modèles."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4
from time import perf_counter

import cv2
import numpy as np

from src.inference.paper_2021 import read_image
from src.paper_reproduction.settings import CLASSES
from src.inference.preprocessing import SHADES, ADAPTIVE

UNVERIFIED = 'Partition non vérifiée : résultat de démonstration uniquement'
MODELS = ['vgg', 'deep', 'fused']


def isic_id(value):
    """Accepter un identifiant exact, éventuellement suivi d'une extension image."""
    name = Path(str(value).strip().replace('\\', '/')).name
    match = re.fullmatch(r'(ISIC_\d{7})(?:\.(?:jpg|jpeg|png))?', name, re.IGNORECASE)
    if not match:
        raise ValueError(f'Identifiant ISIC invalide : {value!r}')
    return match.group(1).upper()


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'CSV introuvable : {path}')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f'Colonnes CSV en doublon : {path}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'CSV sans annotations : {path}')
    return rows


def image_index(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Dossier introuvable : {directory}')
    found = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            identifier = isic_id(path.name)
            if identifier in found:
                raise ValueError(f'Image en doublon pour {identifier} : {found[identifier].name}, {path.name}')
            found[identifier] = path
    return found


def pair_segmentation(images, masks):
    images = image_index(images)
    directory = Path(masks)
    if not directory.is_dir():
        raise FileNotFoundError(f'Dossier de masques introuvable : {directory}')
    indexed = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() == '.png':
            match = re.fullmatch(r'(ISIC_\d{7})_segmentation\.png', path.name, re.IGNORECASE)
            if not match:
                raise ValueError(f'Nom de masque invalide : {path.name}')
            identifier = match.group(1).upper()
            if identifier in indexed:
                raise ValueError(f'Masque en doublon pour {identifier}')
            indexed[identifier] = path
    issues = [f'Masque absent : {i}' for i in sorted(images.keys() - indexed.keys())]
    issues += [f'Image absente : {i}' for i in sorted(indexed.keys() - images.keys())]
    return {i: {'image': images[i], 'mask': indexed[i]} for i in sorted(images.keys() & indexed.keys())}, issues


def read_task3_labels(path):
    rows = read_rows(path)
    key = 'image' if 'image' in rows[0] else 'image_id'
    if not {key, *CLASSES} <= rows[0].keys():
        raise ValueError('CSV Task 3 attendu : image (ou image_id) et MEL, NV, BCC, AKIEC, BKL, DF, VASC.')
    labels = {}
    for row in rows:
        identifier = isic_id(row[key])
        if identifier in labels:
            raise ValueError(f'Étiquette en doublon : {identifier}')
        try:
            values = np.array([float(row[c]) for c in CLASSES])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Annotation one-hot invalide : {identifier}') from exc
        if not np.isin(values, [0., 1.]).all() or values.sum() != 1:
            raise ValueError(f'Annotation one-hot invalide : {identifier}, une seule classe doit valoir 1.')
        labels[identifier] = CLASSES[int(values.argmax())]
    return labels


def classification_records(images, annotations):
    images, labels = image_index(images), read_task3_labels(annotations)
    issues = [f'Annotation absente : {i}' for i in sorted(images.keys() - labels.keys())]
    return {i: {'image': images[i], 'label': labels[i], 'annotation_csv': Path(annotations)}
            for i in sorted(images.keys() & labels.keys())}, issues


def _binary(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0 or not np.isin(mask, [0, 1]).all():
        raise ValueError('Masque binaire 2D non vide attendu, valeurs 0/1.')
    return mask.astype(bool)


def segmentation_metrics(prediction, truth):
    """Scores sans lissage ; deux masques vides donnent Dice=IoU=1."""
    p, g = _binary(prediction), _binary(truth)
    if p.shape != g.shape:
        raise ValueError('Masques non alignés : dimensions différentes.')
    tp, fp = int((p & g).sum()), int((p & ~g).sum())
    fn, tn = int((~p & g).sum()), int((~p & ~g).sum())
    return {'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def load_ground_truth(path, native_hw):
    """Lire un PNG binaire sans EXIF ; alignement par nearest-neighbor uniquement.

    Le nom et les dimensions ne prouvent pas l'orientation anatomique : aucune
    rotation ni inscription spatiale automatique n'est appliquée.
    """
    path = Path(path)
    from PIL import Image
    with Image.open(path) as image:
        if image.format != 'PNG' or image.width * image.height > 40_000_000:
            raise ValueError(f'Masque PNG invalide ou trop grand : {path}')
        image.verify()
    mask = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f'Masque GT binaire attendu (0/1 ou 0/255) : {path}')
    binary = (mask > 0).astype(np.uint8)
    resized = binary.shape != tuple(native_hw)
    if resized:
        # Refuser un rapport de forme différent : un étirement masquerait un défaut d'alignement.
        if abs(binary.shape[1] / binary.shape[0] - native_hw[1] / native_hw[0]) > .01:
            raise ValueError(f'Rapport de forme GT/image incompatible : {path.name}')
        binary = cv2.resize(binary, (native_hw[1], native_hw[0]), interpolation=cv2.INTER_NEAREST)
    return binary, resized


def error_overlay(rgb, prediction, truth):
    segmentation_metrics(prediction, truth)  # vérifier les contrats avant l'affichage
    if rgb.shape[:2] != prediction.shape:
        raise ValueError('Image et masques non alignés.')
    p, g = prediction.astype(bool), truth.astype(bool)
    overlay = rgb.copy()
    for selected, color in [(p & g, [0, 255, 0]), (p & ~g, [255, 0, 0]), (~p & g, [0, 0, 255])]:
        overlay[selected] = (.35 * rgb[selected] + .65 * np.array(color)).astype(np.uint8)
    return overlay


def audit_partition(identifiers, split_path, metadata_path=None):
    """Audit de provenance ; une partition 'test' seule ne prouve pas l'absence de fuite par lésion."""
    identifiers = list(identifiers)
    report = {'split_file': str(split_path), 'all_test': False, 'generalization_eligible': False,
              'message': UNVERIFIED, 'membership': {}, 'duplicate_ids': [], 'lesion_leaks': [],
              'metadata_file': str(metadata_path) if metadata_path else None,
              'metadata_status': 'Regroupements par lésion non vérifiés : métadonnées absentes.'}
    if not Path(split_path).is_file():
        return report
    rows = read_rows(split_path)
    if not {'image_id', 'split'} <= rows[0].keys():
        raise ValueError('Partitions : colonnes image_id et split requises.')
    partitions = defaultdict(set)
    counts = Counter()
    for row in rows:
        identifier, split = isic_id(row['image_id']), row['split'].strip().lower()
        split = {'validation': 'val', 'training': 'train', 'testing': 'test'}.get(split, split)
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'Partition inconnue : {split!r}')
        partitions[identifier].add(split)
        counts[identifier] += 1
    report['split_sha256'] = hashlib.sha256(Path(split_path).read_bytes()).hexdigest()
    report['duplicate_ids'] = sorted(i for i, n in counts.items() if n > 1)
    report['membership'] = {i: sorted(partitions.get(i, {'unknown'})) for i in identifiers}
    report['all_test'] = all(partitions.get(i) == {'test'} and counts[i] == 1 for i in identifiers) and bool(identifiers)
    report['message'] = ('Identifiants vérifiés dans la partition test exportée ; sous-ensemble local.' if report['all_test']
                         else 'Démonstration uniquement : sélection train/validation/inconnue ou partitions ambiguës. Aucune performance finale de généralisation.')
    if metadata_path:
        metadata = read_rows(metadata_path)
        key = 'image_id' if 'image_id' in metadata[0] else 'image'
        if not {key, 'lesion_id'} <= metadata[0].keys():
            raise ValueError('Métadonnées : image_id (ou image) et lesion_id requis.')
        lesions, by_image = defaultdict(set), {}
        for row in metadata:
            identifier, lesion = isic_id(row[key]), row['lesion_id'].strip()
            if identifier in by_image:
                raise ValueError(f'Métadonnée image en doublon : {identifier}')
            by_image[identifier] = lesion
            if lesion:
                lesions[lesion].update(partitions.get(identifier, set()))
        report['metadata_sha256'] = hashlib.sha256(Path(metadata_path).read_bytes()).hexdigest()
        report['lesion_leaks'] = [i for i in identifiers if by_image.get(i) and lesions[by_image[i]] & {'train', 'val'}]
        report['missing_lesion_ids'] = [i for i in partitions if not by_image.get(i)]
        complete = not report['missing_lesion_ids']
        report['metadata_status'] = 'Regroupements par lésion vérifiés.' if complete else 'Métadonnées incomplètes : absence de fuite non garantie.'
        report['generalization_eligible'] = report['all_test'] and complete and not report['lesion_leaks']
        if report['lesion_leaks']:
            report['message'] = 'Fuite potentielle par lésion vers train/validation : démonstration uniquement, aucune performance finale de généralisation.'
    return report


def classification_summary(truth, predictions):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    if not truth or len(truth) != len(predictions) or any(c not in CLASSES for c in [*truth, *predictions]):
        raise ValueError('Classes ou dimensions incompatibles pour le rapport.')
    report = classification_report(truth, predictions, labels=CLASSES, output_dict=True, zero_division=0)
    # Balanced accuracy = rappel moyen des classes réellement présentes.
    present = [c for c in CLASSES if c in truth]
    return {'accuracy': float(accuracy_score(truth, predictions)),
            'balanced_accuracy': float(np.mean([report[c]['recall'] for c in present])),
            'macro_f1': float(report['macro avg']['f1-score']), 'macro_f1_labels': CLASSES,
            'balanced_accuracy_labels_present': present, 'report': report,
            'confusion_matrix': confusion_matrix(truth, predictions, labels=CLASSES).tolist(),
            'class_order': CLASSES}


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def _png(path, image, rgb=False):
    data = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image
    ok, encoded = cv2.imencode('.png', data)
    if not ok:
        raise ValueError(f'Impossible de sauvegarder {path}')
    Path(path).write_bytes(encoded.tobytes())


def evaluate_batch(task, records, pipeline, output, split_path, metadata_path=None, preprocessing=None, adaptive=None):
    """Inférence RGB seule, puis comparaison GT ; exports incrémentaux sans garder le lot en RAM.

    Un lot interrompu porte status=failed et ne produit pas de résumé de réussite.
    Les scores sont ceux du sous-ensemble sélectionné, jamais ceux du test complet.
    """
    if task not in {'segmentation', 'classification'} or not records:
        raise ValueError('Sélection vide ou tâche inconnue.')
    for identifier, record in records.items():
        if isic_id(record['image'].name) != identifier:
            raise ValueError(f'Image/identifiant incohérents : {identifier}')
        if task == 'segmentation' and record['mask'].name.lower() != f'{identifier}_segmentation.png'.lower():
            raise ValueError(f'Image/masque incohérents : {identifier}')
    audit = audit_partition(records, split_path, metadata_path)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex
    directory = Path(output) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'task': task, 'preprocessing_method': preprocessing or SHADES,
                'status': 'running', 'selected': list(records), 'audit': audit,
                'scope': 'Sous-ensemble local sélectionné, pas une performance finale du test complet.',
                'empty_masks_convention': 'Both empty: Dice=IoU=1; no smoothing',
                'completed': [], 'duplicate_pixels': [], 'inputs': {}}
    _json(directory / 'manifest.json', manifest)
    metrics, previews, fingerprints = [], {}, {}
    try:
        for identifier, record in records.items():
            image_start = perf_counter()
            image_path = Path(record['image'])
            image_bytes = image_path.read_bytes()
            if task == 'segmentation':
                # OpenCV tient compte des EXIF ; le GT PNG n'en tient pas compte.
                # Refuser une orientation implicite évite de comparer des repères différents.
                from PIL import Image
                import io
                with Image.open(io.BytesIO(image_bytes)) as source_image:
                    if source_image.getexif().get(274, 1) != 1:
                        raise ValueError(f'Orientation EXIF non neutre : {identifier}. Fournir image et GT dans le même repère natif.')
            rgb = read_image(image_bytes, image_path.name)
            digest = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()
            if digest in fingerprints:
                manifest['duplicate_pixels'].append([fingerprints[digest], identifier])
                audit['generalization_eligible'] = False
                audit['message'] = 'Images identiques dans la sélection : démonstration uniquement, scores pondérés par les doublons.'
            fingerprints[digest] = identifier
            provenance = {'image': str(image_path), 'image_sha256': hashlib.sha256(image_bytes).hexdigest()}
            if task == 'segmentation':
                gt, resized = load_ground_truth(record['mask'], rgb.shape[:2])
                provenance.update(mask=str(record['mask']), mask_sha256=hashlib.sha256(Path(record['mask']).read_bytes()).hexdigest(), gt_resized=resized)
            else:
                if record['label'] not in CLASSES:
                    raise ValueError(f'Classe GT inconnue : {identifier}')
                provenance['true_class'] = record['label']
                if 'annotation_csv' in record:
                    annotation_path = Path(record['annotation_csv'])
                    provenance.update(annotation_csv=str(annotation_path), annotation_csv_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest())
            manifest['inputs'][identifier] = provenance
            # Seuls RGB et le nom sont passés à l'inférence ; jamais le GT.
            result = (pipeline.predict(rgb, image_path.name) if preprocessing is None else
                      pipeline.predict(rgb, image_path.name, preprocessing=preprocessing, adaptive=adaptive))
            execution_seconds = perf_counter() - image_start
            trace = {'preprocessing_method': result.metadata.get('preprocessing_method', preprocessing or SHADES),
                     'steps_executed': json.dumps(result.metadata.get('steps_executed', []), ensure_ascii=False),
                     'model_files': json.dumps(result.metadata.get('model_files', {}), ensure_ascii=False),
                     'preprocessing_seconds': result.metadata.get('preprocessing_seconds'),
                     'execution_seconds': execution_seconds}
            folder = directory / identifier
            folder.mkdir()
            _png(folder / 'original.png', rgb, rgb=True)
            _json(folder / 'inference.json', {**result.metadata, 'execution_seconds': execution_seconds})
            if hasattr(result, 'images') and 'preprocessed' in result.images:
                _png(folder / 'preprocessed.png', result.images['preprocessed'], rgb=True)
            if task == 'segmentation':
                _png(folder / 'ground_truth.png', gt * 255)
                for model in MODELS:
                    prediction = result.images[f'{model}_mask']
                    score = segmentation_metrics(prediction, gt)
                    metrics.append({'image_id': identifier, 'model': model, 'gt_resized': resized, **trace, **score})
                    _png(folder / f'{model}_mask.png', prediction * 255)
                    _png(folder / f'{model}_errors.png', error_overlay(rgb, prediction, gt), rgb=True)
            else:
                predicted = result.metadata['predicted_class']
                if predicted not in CLASSES:
                    raise ValueError(f'Classe prédite inconnue : {predicted}')
                metrics.append({'image_id': identifier, 'true_class': record['label'], 'predicted_class': predicted,
                                'correct': predicted == record['label'], **trace})
            previews[identifier] = str(folder)
            manifest['completed'].append(identifier)
            _csv(directory / 'metrics.csv', metrics)
            _json(directory / 'manifest.json', manifest)
        summary = {'task': task, 'count': len(records), 'audit': audit, 'scope': manifest['scope'],
                   'preprocessing_method': preprocessing or SHADES}
        if task == 'segmentation':
            summary['means'] = {model: {metric: float(np.mean([r[metric] for r in metrics if r['model'] == model]))
                                        for metric in ['dice', 'iou']} for model in MODELS}
        else:
            summary.update(classification_summary([r['true_class'] for r in metrics], [r['predicted_class'] for r in metrics]))
            _csv(directory / 'classification_report.csv', [{'class': c, **summary['report'][c]} for c in CLASSES])
            _csv(directory / 'confusion_matrix.csv', [{'true_class': c, **dict(zip(CLASSES, row))}
                                                      for c, row in zip(CLASSES, summary['confusion_matrix'])])
        manifest['status'] = 'complete'
        _json(directory / 'summary.json', summary)
        _json(directory / 'manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        _json(directory / 'manifest.json', manifest)
        raise RuntimeError(f'Évaluation interrompue ; résultats partiels marqués failed dans {directory} : {exc}') from exc
    return {'directory': directory, 'summary': summary, 'metrics': metrics, 'previews': previews}


def selected_image_ids(path):
    """Lire la sélection fournie sans utiliser sa colonne split comme preuve de partition."""
    rows = read_rows(path)
    if 'image_id' not in rows[0]:
        raise ValueError('Sélection : colonne image_id requise.')
    ids = [isic_id(row['image_id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Identifiants en doublon dans selected_images.csv.')
    return ids


def compare_summaries(shades, adaptive):
    """Différence explicite adaptatif moins Shades, sans supposer son signe."""
    if shades['task'] != adaptive['task'] or shades['count'] != adaptive['count']:
        raise ValueError('Comparaison impossible : tâches ou tailles de lots différentes.')
    rows = []
    if shades['task'] == 'segmentation':
        for model in MODELS:
            for metric in ['dice', 'iou']:
                a, b = shades['means'][model][metric], adaptive['means'][model][metric]
                rows.append({'model': model, 'metric': metric, SHADES: a, ADAPTIVE: b, 'adaptive_minus_shades': b - a})
    else:
        for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']:
            a, b = shades[metric], adaptive[metric]
            rows.append({'metric': metric, SHADES: a, ADAPTIVE: b, 'adaptive_minus_shades': b - a})
    return rows


def evaluate_comparison(task, records, pipeline, adaptive, output, split_path, metadata_path=None):
    """Même instance de modèles, mêmes identifiants et vérification des empreintes d'entrée."""
    if not records:
        raise ValueError('Sélection vide.')
    audit = audit_partition(records, split_path, metadata_path)
    directory = Path(output) / ('comparison_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'task': task, 'selected': list(records), 'audit': audit, 'methods': [SHADES, ADAPTIVE]}
    _json(directory / 'manifest.json', manifest)
    start = perf_counter()
    try:
        runs = {}
        for method in [SHADES, ADAPTIVE]:
            runs[method] = evaluate_batch(task, records, pipeline, directory / method, split_path, metadata_path,
                                          preprocessing=method, adaptive=adaptive if method == ADAPTIVE else None)
        manifests = [json.loads((runs[m]['directory'] / 'manifest.json').read_text(encoding='utf-8')) for m in [SHADES, ADAPTIVE]]
        if manifests[0]['inputs'] != manifests[1]['inputs'] or manifests[0]['audit'] != manifests[1]['audit']:
            raise ValueError('Les images, annotations ou partitions ont changé entre les méthodes : comparaison refusée.')
        comparison = compare_summaries(runs[SHADES]['summary'], runs[ADAPTIVE]['summary'])
        manifest.update(status='complete', audit=manifests[0]['audit'], execution_seconds=perf_counter() - start,
                        runs={m: str(r['directory']) for m, r in runs.items()})
        _csv(directory / 'comparison.csv', comparison)
        _json(directory / 'summary.json', {'comparison': comparison, **manifest})
        _json(directory / 'manifest.json', manifest)
        return {'directory': directory, 'comparison': comparison, 'runs': runs}
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        _json(directory / 'manifest.json', manifest)
        raise RuntimeError(f'Comparaison interrompue, lot marqué failed : {directory} : {exc}') from exc
