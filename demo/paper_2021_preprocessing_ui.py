"""Deux boutons exclusifs ; le choix ne déclenche pas l'inférence."""
import streamlit as st
from src.inference.preprocessing import SHADES, ADAPTIVE, METHOD_LABELS, AdaptivePreprocessor


@st.cache_resource(show_spinner=False)
def get_adaptive(project):
    return AdaptivePreprocessor(project)


def _choose(method):
    st.session_state['paper_preprocessing_method'] = method
    for key in ['paper_result', 'evaluation_result']:
        st.session_state.pop(key, None)


def preprocessing_buttons():
    st.session_state.setdefault('paper_preprocessing_method', SHADES)
    method = st.session_state['paper_preprocessing_method']
    st.subheader('Méthode de prétraitement')
    for column, key in zip(st.columns(2), [SHADES, ADAPTIVE]):
        column.button(METHOD_LABELS[key], type='primary' if method == key else 'secondary',
                      on_click=_choose, args=(key,), key=f'choose_{key}', use_container_width=True)
    st.caption(f'Méthode sélectionnée : {METHOD_LABELS[method]}')
    if method == ADAPTIVE:
        st.info('C1 automatique. C2–C5 : détections signalées ; corrections C2/C3/C4 désactivées et C5 sans correction disponible.')
    return method
