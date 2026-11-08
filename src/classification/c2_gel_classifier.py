from pathlib import Path
import json
import joblib
import numpy as np
from .c2_features import extract_c2_features, FEATURE_NAMES
from .c2_reflection_detector import detect_reflections

class C2Detector:
    def __init__(self, project_path):
        root=Path(project_path)
        cfg=json.loads((root/'models/c2_config.json').read_text(encoding='utf-8'))
        if list(cfg['feature_names'])!=list(FEATURE_NAMES):
            raise ValueError('Feature schema incompatible avec les modèles sauvegardés')
        self.models={}
        self.thresholds={}
        for label,record in cfg['labels'].items():
            self.models[label]=joblib.load(root/'models'/record['filename'])
            self.thresholds[label]=float(record['threshold'])
        p=root/'models/c2_reflections_config.json'
        self.reflection_config=json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

    def predict(self,image_rgb):
        features=extract_c2_features(image_rgb).reshape(1,-1)
        gel={}
        for key,model in self.models.items():
            prob=float(model.predict_proba(features)[0,1])
            gel[key]={'score':prob,'threshold':self.thresholds[key],'detected':bool(prob>=self.thresholds[key])}
        reflections=detect_reflections(image_rgb,self.reflection_config)
        c2=any(v['detected'] for v in gel.values()) or reflections['reflection_detected']
        return {'C2':bool(c2),'gel':gel,'reflection_detected':reflections['reflection_detected'],
                'reflection_mask':reflections['mask'],'reflection_area_ratio':reflections['area_ratio'],
                'reflection_safe_to_inpaint':reflections['safe_to_inpaint']}
