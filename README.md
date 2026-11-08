# Adaptive Skin Lesion Analysis

**Deep Learning-Based Skin Lesion Segmentation and Multiclass Classification Using ISIC 2018**

## Overview

This project explores an automated skin lesion analysis pipeline using deep learning and hybrid machine learning techniques.

The system combines adaptive image preprocessing, lesion segmentation, handcrafted feature extraction, deep feature learning, and multiclass classification.

The experiments are conducted using the **ISIC 2018 dataset**, with the objective of reproducing and evaluating methods described in scientific literature.

## Methodology

The pipeline consists of five main stages:

1. **Adaptive Image Preprocessing:** Image analysis and categorization according to visual characteristics and artifacts, including hair detection. Appropriate preprocessing operations are selected based on the detected image characteristics, alongside color normalization using Shades of Gray.

2. **Lesion Segmentation:** Skin lesion segmentation using VGG19-UNet and DeepLabV3+, followed by segmentation ensemble.

3. **Feature Extraction:** Extraction of 200 handcrafted features combined with deep CNN embeddings.

4. **Multiclass Classification:** Hybrid ResNet50, DenseNet201, and Support Vector Machine (SVM) for skin lesion classification.

5. **Evaluation:** Performance assessment using Dice, IoU, accuracy, F1-score, and other classification metrics.

## Technologies

Python · PyTorch · OpenCV · Scikit-learn · NumPy · Pandas · Google Colab

## Experimental Results

Segmentation and classification experiments are evaluated separately. Results are reported with their dataset split, evaluation protocol, and model configuration.

*Final evaluation results will be added after completion of the experiments.*

## Disclaimer

This is an academic research project and is not intended for clinical diagnosis.
