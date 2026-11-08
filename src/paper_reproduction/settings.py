"""Independent ISIC 2018 experimental implementation; Ali & Ragb (2021) inspired."""
from pathlib import Path
import json, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
