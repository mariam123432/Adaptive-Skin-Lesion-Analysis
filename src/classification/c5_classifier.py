"""Load trained C5 classifiers. Does not retrain or depend on Bissoto labels."""
from pathlib import Path
import json
import joblib
import numpy as np
from .c5_features import FEATURE_NAMES, extract_c5_features

class C5Detector:
    def __init__(self, project_path):
        self.root=Path(project_path)
        config_path=self.root/'models'/'c5_config.json'
        with open(config_path,encoding='utf-8') as f:
            self.config=json.load(f)
        if self.config['feature_names'] != FEATURE_NAMES:
            raise ValueError('Feature order mismatch: verify c5_features.py and model config')
        self.models={}
        for label in ('ruler','dark_corner'):
            file=self.root/'models'/self.config['labels'][label]['file']
            self.models[label]=joblib.load(file)
    def predict(self,image_rgb):
        x=extract_c5_features(image_rgb).reshape(1,-1)
        output={}
        for label,model in self.models.items():
            score=float(model.predict_proba(x)[0,1])
            threshold=float(self.config['labels'][label]['threshold'])
            output[label]={'score':score,'threshold':threshold,'detected':bool(score>=threshold)}
        output['C5']=any(output[label]['detected'] for label in ('ruler','dark_corner'))
        return output
