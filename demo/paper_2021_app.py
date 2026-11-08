"""Démonstration Streamlit séparée du prétraitement adaptatif C1–C5."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title='Lésions cutanées — pipeline 2021', layout='wide')
st.title('Segmentation et classification des lésions cutanées')
st.caption('Adaptation expérimentale de « Skin Lesion Segmentation and Classification Using Deep Learning and Handcrafted Features ».')
st.warning('Prototype expérimental de recherche — ne constitue pas un diagnostic médical.')

try:
    import torch
    from src.inference.paper_2021 import Paper2021Pipeline, EXTENSIONS, read_image
except ImportError as exc:
    st.error(f'Dépendance indisponible : {exc}. Utilisez le Python du dossier .venv (voir le guide).')
    st.stop()


@st.cache_resource(show_spinner=False)
def get_pipeline(device):
    """Une seule instance par choix de device, partagée entre les sessions."""
    return Paper2021Pipeline(ROOT, device)


from demo.paper_2021_preprocessing_ui import preprocessing_buttons, get_adaptive
from src.inference.preprocessing import SHADES, ADAPTIVE
with st.sidebar:
    preprocessing_method = preprocessing_buttons()

activity = st.sidebar.radio('Mode', ['Démonstration', 'Évaluation Ground Truth'])
if activity == 'Évaluation Ground Truth':
    from demo.paper_2021_evaluation_ui import render_evaluation
    render_evaluation(ROOT, get_pipeline, preprocessing_method, get_adaptive)
    st.stop()


with st.sidebar:
    st.header('Image à analyser')
    mode = st.selectbox('Source', ['Dossier test_images', 'Importer une image'])
    device = st.selectbox('Calcul', ['auto', 'cpu', 'cuda'])
    st.caption(f'CUDA disponible : {torch.cuda.is_available()}')
    data, filename = None, None
    if mode == 'Dossier test_images':
        folder = ROOT / 'test_images'
        # Limiter la sélection au dossier prévu et aux extensions autorisées.
        files = sorted(p for p in folder.iterdir() if p.is_file() and
                       p.suffix.lower() in EXTENSIONS and p.resolve().parent == folder.resolve()) if folder.is_dir() else []
        if files:
            selected = st.selectbox('Image locale', files, format_func=lambda p: p.name)
            filename = selected.name
            try:
                if selected.stat().st_size > 30 * 1024 * 1024:
                    raise ValueError('Image supérieure à 30 Mo.')
                data = selected.read_bytes()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        else:
            st.info('Ajoutez des images .jpg, .jpeg ou .png dans test_images/, ou utilisez l’import.')
    else:
        uploaded = st.file_uploader('Image dermoscopique', type=['jpg', 'jpeg', 'png'])
        if uploaded is not None:
            filename, data = uploaded.name, uploaded.getvalue()
    analyze = st.button('Analyser l’image', type='primary', disabled=data is None)
    st.caption('Les modèles sont chargés au premier clic. Aucun entraînement ni téléchargement de poids.')

# Empêcher l'affichage d'une ancienne prédiction sous le nom d'une nouvelle image.
identity = (hashlib.sha256(data).hexdigest(), filename, device, mode, preprocessing_method) if data is not None else None
if st.session_state.get('paper_identity') != identity:
    st.session_state.pop('paper_result', None)
    st.session_state['paper_identity'] = identity

if analyze:
    st.session_state.pop('paper_result', None)
    try:
        with st.spinner('Validation de l’image, chargement des modèles et inférence…'):
            rgb = read_image(data, filename)
            adaptive = get_adaptive(str(ROOT)) if preprocessing_method == ADAPTIVE else None
            pipeline = get_pipeline(device)
            st.session_state['paper_result'] = pipeline.predict(rgb, filename, preprocessing=preprocessing_method, adaptive=adaptive)
    except Exception as exc:
        # Une incompatibilité doit rester visible, sans résultat de substitution.
        st.error(f'Analyse interrompue : {exc}')

overview, preprocessing, segmentation, classification, technical, export = st.tabs(
    ['Vue générale', 'Prétraitement', 'Segmentation', 'Classification', 'Détails techniques', 'Export'])

with overview:
    st.write('Image RGB → méthode de prétraitement sélectionnée → image 224 × 224 → normalisation ImageNet.')
    st.write('VGG19-UNet + DeepLabV3+ → moyenne des probabilités → seuil 0,5 → masque et superposition.')
    st.write('200 caractéristiques manuelles → scaler → CNN hybrides ResNet50 et DenseNet201 → deux embeddings de 512 → SVM (1024 valeurs).')
    st.info('Le projet utilise un descripteur handcrafted200 adapté. Cette démonstration ne prouve pas une reproduction exacte des résultats du papier.')

result = st.session_state.get('paper_result')
if result is None:
    with overview:
        st.info('Sélectionnez une image puis cliquez sur « Analyser l’image ».')
    st.stop()

with preprocessing:
    left, right = st.columns(2)
    left.image(result.images['original'], caption='Image originale RGB', use_container_width=True)
    right.image(result.images['preprocessed'], caption='Image réellement utilisée avant normalisation ImageNet (224 × 224)', use_container_width=True)
    st.write('Méthode :', result.metadata['preprocessing_label'])
    st.write('Étapes exécutées :', result.metadata['steps_executed'])
    st.metric('Temps de prétraitement (s)', f"{result.metadata['preprocessing_seconds']:.4f}")
    if 'adaptive_report' in result.metadata:
        st.json(result.metadata['adaptive_report'])
        st.warning(result.metadata['adaptive_policy'])
    st.caption('Le tenseur transmis aux réseaux reçoit ensuite la normalisation ImageNet du projet.')

with segmentation:
    cols = st.columns(3)
    for col, key, title in zip(cols, ['vgg', 'deep', 'fused'], ['VGG19-UNet', 'DeepLabV3+', 'Fusion par moyenne']):
        col.image(result.images[f'{key}_mask'] * 255, caption=f'Masque {title}', use_container_width=True)
    st.image(result.images['overlay'], caption='Masque final superposé sur l’image originale', use_container_width=True)
    with st.expander('Cartes de probabilités à 224 × 224'):
        for col, key, title in zip(st.columns(3), ['vgg', 'deep', 'fused'], ['VGG19-UNet', 'DeepLabV3+', 'Moyenne']):
            col.image(result.probabilities[key], caption=title, clamp=True, use_container_width=True)
    if result.metadata['empty_mask']:
        st.warning('Le masque final est vide. Le descripteur applique le comportement de secours du code scientifique existant ; interprétez la classe avec prudence.')
    st.caption('Sans annotation de référence, aucun Dice, IoU ou taux de réussite n’est calculé.')

with classification:
    st.metric('Catégorie prédite', result.metadata['predicted_class'])
    st.write('CNN hybrides : ResNet50 puis DenseNet201.')
    a, b, c = st.columns(3)
    a.metric('Caractéristiques manuelles', 200)
    b.metric('Chaque embedding hybride', 512)
    c.metric('Entrée du SVM', 1024)
    st.caption('La classe est une sortie expérimentale du SVM. Aucun score n’est présenté comme une probabilité médicale.')
    st.write('Classes : MEL, NV, BCC, AKIEC, BKL, DF, VASC.')
    with st.expander('Les 200 valeurs manuelles avant scaler'):
        st.json(result.manual_features.tolist())

with technical:
    st.metric('Device', result.metadata['device'])
    st.json(result.metadata)
    st.caption('Les temps d’inférence excluent le chargement initial, indiqué séparément. En CUDA, les mesures sont synchronisées.')

with export:
    # Ces exports ne relancent jamais les réseaux : le résultat reste en session.
    files = result.files()
    st.download_button('Télécharger l’exécution complète (ZIP)', result.zip_bytes(), 'paper_2021_run.zip', 'application/zip')
    st.download_button('Télécharger prediction.json', files['prediction.json'], 'prediction.json', 'application/json')
    for name, content in files.items():
        if name.endswith('.png'):
            st.download_button(f'Télécharger {name}', content, name, 'image/png', key=f'download_{name}')
    if st.button('Sauvegarder cette exécution sur disque'):
        try:
            destination = result.save(ROOT / 'results/paper_2021/demo_runs')
            st.success(f'Exécution sauvegardée : {destination}')
        except (OSError, ValueError) as exc:
            st.error(f'Échec de sauvegarde : {exc}')
