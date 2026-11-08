import numpy as np
from src.classification.hair_features import FEATURE_NAMES, extract_hair_features
from src.classification.quality import classify_problems
from src.pipeline import process_image

def test_without_model():
    rgb=np.full((120,160,3),160,dtype=np.uint8)
    x=extract_hair_features(rgb)
    assert len(FEATURE_NAMES)==14 and x.shape==(14,)
    result=process_image(rgb)
    assert result["classes"]["C1"] is None
    assert np.array_equal(result["cleaned_image"],rgb)
