"""Usage : python -m scripts.predict_image chemin/vers/image.jpg --model 'SVM RBF'"""
import argparse, json
from pathlib import Path
import cv2
from src.classification.hair_predictor import HairPredictor
from src.pipeline import process_image

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", choices=["SVM RBF", "Random Forest"], default="SVM RBF")
    ap.add_argument("--output", type=Path, default=Path("outputs/cleaned/demo_cleaned.png"))
    args=ap.parse_args()
    bgr=cv2.imread(str(args.image))
    if bgr is None: raise FileNotFoundError(args.image)
    image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor=HairPredictor(Path(__file__).resolve().parents[1], args.model)
    result=process_image(image, predictor)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(args.output),cv2.cvtColor(result["cleaned_image"],cv2.COLOR_RGB2BGR))
    print(json.dumps({k:v for k,v in result.items() if k!="cleaned_image"},ensure_ascii=False,indent=2))
    print("Image sauvegardée :", args.output)
if __name__ == "__main__": main()
