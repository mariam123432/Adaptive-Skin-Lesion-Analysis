"""Stable public API; implementation split by task for local reuse."""
from .settings import CLASSES
from .preprocessing import IMAGENET_MEAN, IMAGENET_STD,normalize_tensor, shades_of_gray
from .data import seed_all,read_rgb,image_lookup,mask_lookup,build_pairs,SegDataset,DiagnosisDataset
from .segmentation import dice_loss,seg_loss,seg_metrics,make_segmenter,train_seg,ensemble_mask
from .features import handcrafted200
from .diagnosis import HybridCNN,train_hybrid,extract_embeddings
