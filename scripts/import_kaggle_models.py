"""Importer les fichiers entraînés copiés depuis les outputs de Kaggle.
Usage : python -m scripts.import_kaggle_models /chemin/vers/export_Kaggle
"""
import shutil, sys
from pathlib import Path

def main(source):
    root=Path(__file__).resolve().parents[1]
    source=Path(source)
    for name in ["hair_svm_rbf.joblib","hair_random_forest.joblib","hair_config.json"]:
        p=next((q for q in [source/name,source/"models"/name,source/"ISIC2018_pipeline_adaptatif"/"models"/name] if q.is_file()),None)
        if p is not None:
            (root/"models").mkdir(exist_ok=True)
            shutil.copy2(p,root/"models"/name)
            print("Importé",name)
        else:
            print("Absent",name)
if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("Fournir le dossier exporté depuis Kaggle")
    main(sys.argv[1])
