"""Démonstration locale C1 (Streamlit)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from src.classification.hair_predictor import HairPredictor
from src.pipeline import process_image

st.title("ISIC 2018 — prétraitement adaptatif")
st.caption("Prototype : C1 opérationnel avec poids exportés ; C2–C5 à développer.")
root=Path(__file__).resolve().parents[1]
choice=st.selectbox("Modèle C1",["SVM RBF","Random Forest"])
file=st.file_uploader("Image dermoscopique",type=["jpg","jpeg","png"])
if file:
    bgr=cv2.imdecode(np.frombuffer(file.getvalue(),np.uint8),cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Image illisible")
    else:
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        try:
            detector=HairPredictor(root,choice)
            output=process_image(rgb,detector)
            st.json({"classes":output["classes"],"details":output["details"],"traitements":output["applied"]})
            a,b=st.columns(2)
            a.image(rgb,caption="Originale",use_container_width=True)
            b.image(output["cleaned_image"],caption="Après nettoyage conditionnel",use_container_width=True)
        except (FileNotFoundError,ValueError) as e:
            st.error(str(e))
