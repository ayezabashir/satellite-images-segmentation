# Semantic Segmentation using Deep Learning Models

This repository contains deep learning implementations of semantic segmentation using satellite imagery. The project focuses on multi-class pixel-wise classification into four distinct environmental categories: **Crops**, **Bare Land**, **Built Up**, and **Water**.

Several state-of-the-art vision architectures are explored and compared, including Transformer-based models and classical Convolutional Neural Networks (CNNs).

---

## Project Methodology

The workflow extends from data ingestion via Google Earth Engine to localized patch processing, model training, and performance evaluation in Google Colab.

<img width="1366" height="588" alt="methodology OVERVIEW" src="https://github.com/user-attachments/assets/98419171-c48a-4767-bdca-5c74b818490b" />

1. **Preprocessing (Google Earth Engine):** Define region of interest (ROI), generate cloud-free median composites over the selected time range, and export the primary RGB (Red, Green, Blue) bands.
2. **Dataset Preparation (Python):** Split the wide-area satellite composites and corresponding ground-truth masks into localized $128 \times 128$ pixel patches. These patches are consolidated into structured Numpy arrays (`.npy`) for training, validation, and testing.
3. **Model Training & Evaluation (Google Colab):** Train deep learning architectures utilizing optimized processing configurations (GPU acceleration) and benchmark performance using standard semantic segmentation metrics.

---

## Models Implemented

The repository features dedicated Jupyter Notebooks (`.ipynb`) for training and evaluating four distinct architectures:

*   **`UNet.ipynb`** – The classic convolutional encoder-decoder baseline tailored for biomedical and remote sensing segmentation.
*   **`MobileViT.ipynb`** – A light-weight, mobile-friendly transformer architecture that blends CNN-like inductive biases with global self-attention.
*   **`segformer_b0.ipynb`** – A highly efficient, lightweight variant of the SegFormer architecture (`nvidia/mit-b0`).
*   **`Segformer_b2_.ipynb`** – A larger, more robust variant of the SegFormer pipeline (`nvidia/mit-b2`) utilizing hierarchical Transformer encoders alongside clean All-MLP decoders.

---

## Dataset & Class Distribution

The target masks are indexed pixel-by-pixel into four key LULC classification categories:

| Class Index | Class Name | Color Map Representation |
| :---: | :--- | :--- |
| **0** | Crops | Green (`#00FF00`) |
| **1** | Bare Land | Tan / Beige (`#D2B48C`) |
| **2** | Built Up | Gray (`#808080`) |
| **3** | Water | Blue (`#0000FF`) |

---

## Results & Predictions Comparison

Below is a visual side-by-side evaluation mapping the Ground Truth against predictions generated across the different trained models (**SegFormer B0**, **SegFormer B2**, **SegFormer B4**, **MobileViT**, and **UNet**):

<img width="963" height="687" alt="Predictions" src="https://github.com/user-attachments/assets/8259e6b3-9c2d-4a38-89ca-1ff1d0f196ee" />

### Key Metrics Tracked
During model evaluation on the test subset, the following standard metrics are computed globally and on a per-class basis:
*   **Pixel-wise Accuracy**
*   **Intersection over Union (IoU) / Mean IoU (mIoU)**
*   **Dice Coefficient / F1-Score**
*   **Confusion Matrix (Pixel-level heatmap)**

---

## Author & Developer

This project was designed, developed, and evaluated entirely by [Ayeza Bashir](https://github.com/ayezabashir) as part of my work in remote sensing and deep learning. 
