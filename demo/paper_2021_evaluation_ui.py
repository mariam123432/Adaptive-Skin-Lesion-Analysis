"""Présentation de l'évaluation ; tous les calculs sont dans src.evaluation."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation.paper_2021 import (
    pair_segmentation, classification_records, audit_partition, evaluate_batch,
    selected_image_ids, evaluate_comparison,
)
from src.inference.paper_2021 import Paper2021Segmenter
from src.inference.preprocessing import SHADES, ADAPTIVE, METHOD_LABELS


@st.cache_resource(show_spinner=False)
def get_segmenter(project, device):
    return Paper2021Segmenter(project, device)


def render_evaluation(root, get_pipeline, preprocessing=SHADES, get_adaptive=None):
    st.header('Évaluation Ground Truth')
    st.caption('Annotations locales fournies par l’utilisateur. Leur provenance officielle doit être conservée ; le nom du fichier ne garantit pas son authenticité.')
    with st.sidebar:
        task_label = st.selectbox('Tâche ISIC 2018', ['Segmentation — Task 1', 'Classification — Task 3'])
        device = st.selectbox('Calcul pour l’évaluation', ['auto', 'cpu', 'cuda'])
    task = 'segmentation' if task_label.startswith('Segmentation') else 'classification'
    base = root / 'test_images' / task
    split = root / 'results/paper_2021' / ('task1_splits.csv' if task == 'segmentation' else 'task3_splits.csv')
    with st.expander('Partitions exportées et métadonnées de lésions', expanded=True):
        split_text = st.text_input('CSV des partitions exportées', str(split), key=f'split_{task}')
        metadata_text = st.text_input('CSV de métadonnées (optionnel : image_id, lesion_id)', key=f'metadata_{task}')
        st.caption('Les chemins relatifs sont résolus depuis la racine du projet. Les anciens chemins Kaggle contenus dans les CSV ne sont pas utilisés pour charger les images.')
    split_path = Path(split_text)
    if not split_path.is_absolute():
        split_path = root / split_path
    metadata_path = Path(metadata_text) if metadata_text.strip() else None
    if metadata_path and not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    try:
        if task == 'segmentation':
            st.write('Déposez vos JPG dans `test_images/segmentation/images/` et les PNG correspondants dans `test_images/segmentation/masks/`.')
            records, issues = pair_segmentation(base / 'images', base / 'masks')
        else:
            st.write('Déposez vos images dans `test_images/classification/images/` et remplacez `test_images/classification/ground_truth.csv` par les annotations officielles Task 3.')
            records, issues = classification_records(base / 'images', base / 'ground_truth.csv')
        for issue in issues:
            st.warning(issue)
        if not records:
            st.info('Aucune image annotée disponible. Ajoutez de véritables images et annotations dans les dossiers indiqués.')
            return
        only_test = st.checkbox('Proposer uniquement les identifiants du test exporté', value=False, key=f'only_test_{task}')
        full_audit = audit_partition(records, split_path, metadata_path)
        options = [i for i in records if not only_test or full_audit['membership'].get(i) == ['test']]
        selection_csv = base / 'selected_images.csv'
        if selection_csv.is_file():
            if st.checkbox('Utiliser le sous-ensemble de selected_images.csv', value=True, key=f'use_csv_{task}'):
                listed = selected_image_ids(selection_csv)
                missing = sorted(set(listed) - set(records))
                if missing:
                    st.warning(f'Identifiants du CSV sans image/annotation disponible : {", ".join(missing)}')
                options = [i for i in options if i in set(listed)]
        selected = st.multiselect('Ensemble local à évaluer', options, key=f'selection_{task}')
        if not selected:
            st.info('Sélectionnez une ou plusieurs images. Aucun modèle n’est chargé avant le clic d’évaluation.')
            return
        audit = audit_partition(selected, split_path, metadata_path)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    st.warning(audit['message'])
    st.info(audit['metadata_status'])
    with st.expander('Détail de l’audit avant inférence'):
        st.json(audit)
    st.caption('Même sur des identifiants test, ces scores décrivent le sous-ensemble sélectionné. Ils ne constituent pas les performances finales du test complet.')
    compare = st.checkbox('Comparer les deux méthodes sur exactement les mêmes images', key=f'compare_{task}')

    # Invalider un résultat si la sélection, un fichier d'entrée ou l'audit change.
    watched = [split_path] + ([metadata_path] if metadata_path else [])
    for identifier in selected:
        watched.extend(value for value in records[identifier].values() if isinstance(value, Path))
    if task == 'classification':
        watched.append(base / 'ground_truth.csv')
    watched.append(selection_csv)
    signature = (task, device, preprocessing, compare, tuple(selected), tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                                                     if p.is_file() else (str(p), None, None) for p in watched))
    if st.session_state.get('evaluation_signature') != signature:
        st.session_state.pop('evaluation_result', None)
        st.session_state['evaluation_signature'] = signature
    if st.button('Évaluer la sélection et sauvegarder', type='primary'):
        st.session_state.pop('evaluation_result', None)
        try:
            with st.spinner('Inférence et comparaison aux véritables annotations…'):
                adaptive = get_adaptive(str(root)) if (compare or preprocessing == ADAPTIVE) and get_adaptive else None
                if (compare or preprocessing == ADAPTIVE) and adaptive is None:
                    raise ValueError('Adaptateur de prétraitement adaptatif indisponible.')
                pipeline = get_segmenter(str(root), device) if task == 'segmentation' else get_pipeline(device)
                records_selected = {i: records[i] for i in selected}
                if compare:
                    st.session_state['evaluation_result'] = evaluate_comparison(
                        task, records_selected, pipeline, adaptive,
                        root / 'results/paper_2021/evaluation', split_path, metadata_path)
                else:
                    st.session_state['evaluation_result'] = evaluate_batch(
                        task, records_selected, pipeline,
                        root / 'results/paper_2021/evaluation', split_path, metadata_path,
                        preprocessing=preprocessing, adaptive=adaptive)
        except Exception as exc:
            st.error(f'Évaluation impossible : {exc}')
    run = st.session_state.get('evaluation_result')
    if run is None:
        return
    if 'comparison' in run:
        st.subheader('Comparaison appariée des prétraitements')
        st.dataframe(pd.DataFrame(run['comparison']), use_container_width=True)
        st.caption('Différence = adaptatif − Shades of Gray. Aucune amélioration n’est présumée. Modèles et identifiants identiques.')
        st.download_button('Télécharger comparison.csv', (run['directory'] / 'comparison.csv').read_bytes(), 'comparison.csv', 'text/csv')
        method = st.selectbox('Détails de la méthode', [SHADES, ADAPTIVE], format_func=lambda value: METHOD_LABELS[value])
        run = run['runs'][method]
    summary = run['summary']
    st.write('Méthode évaluée :', METHOD_LABELS[summary['preprocessing_method']])
    st.success(f"Résultats sauvegardés : {run['directory']}")
    st.warning(summary['audit']['message'])
    st.metric('Images évaluées', summary['count'])
    if task == 'segmentation':
        st.dataframe(pd.DataFrame(summary['means']).T.rename_axis('modèle'), use_container_width=True)
        st.caption('Dice et IoU : moyenne arithmétique par image, sans lissage. Deux masques vides donnent 1. Scores à la résolution native, seuil 0,5.')
    else:
        a, b, c = st.columns(3)
        a.metric('Accuracy', f"{summary['accuracy']:.4f}")
        b.metric('Balanced accuracy', f"{summary['balanced_accuracy']:.4f}")
        c.metric('Macro-F1 (7 classes)', f"{summary['macro_f1']:.4f}")
        st.caption('Balanced accuracy : rappel moyen des classes présentes dans le GT. Macro-F1 : sept classes, scores indéfinis fixés à zéro. Aucun score de confiance SVM n’est inventé.')
        st.subheader('Matrice de confusion — lignes réelles, colonnes prédites')
        st.dataframe(pd.DataFrame(summary['confusion_matrix'], index=summary['class_order'], columns=summary['class_order']))
        st.subheader('Rapport de classification : précision, rappel, F1 et support')
        st.dataframe(pd.DataFrame(summary['report']).T)
    st.subheader('Résultats par image')
    st.dataframe(pd.DataFrame(run['metrics']), use_container_width=True)
    chosen = st.selectbox('Visualiser une image évaluée', list(run['previews']))
    folder = Path(run['previews'][chosen])
    st.image(str(folder / 'original.png'), caption=chosen)
    if task == 'segmentation':
        st.image(str(folder / 'ground_truth.png'), caption='Ground Truth aligné à la taille native')
        for col, model, label in zip(st.columns(3), ['vgg', 'deep', 'fused'], ['VGG19-UNet', 'DeepLabV3+', 'Ensemble']):
            col.image(str(folder / f'{model}_mask.png'), caption=label)
            col.image(str(folder / f'{model}_errors.png'), caption='Vert : TP · Rouge : FP · Bleu : FN')
        st.caption('Le GT sert uniquement à la comparaison après prédiction. Vérifiez visuellement l’orientation : aucun recalage anatomique n’est réalisé.')
    else:
        row = next(row for row in run['metrics'] if row['image_id'] == chosen)
        st.write(f"Classe réelle : **{row['true_class']}** · Classe prédite : **{row['predicted_class']}**")
        st.write('Prédiction correcte' if row['correct'] else 'Prédiction incorrecte')
    for name, mime in [('metrics.csv', 'text/csv'), ('summary.json', 'application/json'), ('manifest.json', 'application/json')]:
        st.download_button(f'Télécharger {name}', (run['directory'] / name).read_bytes(), name, mime)
