# Cancer Detection (Brain Tumor MRI Classification)

## Objective

Classify brain MRI scans into 4 categories — glioma, meningioma, pituitary
tumor, or no tumor — using a Convolutional Neural Network.

## Dataset Link

[Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Downloaded automatically by the notebook via the `opendatasets` library
(see setup note below — this requires a one-time Kaggle API credential
setup). 5,600 training images and 1,600 testing images across the 4
classes (1,400 / 400 per class respectively).

## Libraries Used

- `tensorflow` / `keras` — CNN model, training callbacks, data augmentation
- `opendatasets` — downloads the dataset directly from Kaggle
- `matplotlib`, `PIL` — visualizations
- `numpy`

## Methodology

1. Downloaded the Brain Tumor MRI dataset from Kaggle via `opendatasets`.
2. Inspected class distribution across the Training/Testing splits
   (perfectly balanced: 1,400 train / 400 test images per class).
3. Visualized one sample image per class.
4. Built an `ImageDataGenerator` pipeline (rescaling, rotation, shifts,
   horizontal flip, zoom, shear, brightness jitter) for the training set,
   with images resized to 128×128.
5. Built a CNN (convolutional blocks with BatchNorm + Dropout) and trained
   with `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping`.
6. Plotted training vs. validation accuracy/loss curves and reported
   final metrics.

## Results

- **Final Training Accuracy: 98.21%**
- **Best Validation Accuracy: 91.25%**
- Gap between training and validation accuracy: **6.96%** (mild
  overfitting — the model fits training data somewhat more tightly than
  it generalizes, though validation accuracy is still strong)

## Conclusion

The CNN classifies brain MRI scans into the 4 tumor categories with a
best validation accuracy of 91.25%. The ~7-point gap between training and
validation accuracy indicates mild overfitting — expected given the
relatively small, highly specialized medical image dataset (5,600 training
images across 4 classes) and the model's capacity. Techniques already in
use (dropout, batch normalization, data augmentation, early stopping)
helped keep the gap from being larger, but a stricter regularization
schedule, more aggressive augmentation, or transfer learning from a
pretrained medical-imaging backbone could likely close it further. As
with any medical imaging classifier, this model is a proof-of-concept for
a learning exercise — it is not validated for clinical use and would need
far more rigorous evaluation (e.g. per-class sensitivity/specificity,
external validation on a separate hospital's scans) before any real-world
diagnostic application.

---

## Project structure

```
4-cancer-detection/
├── cancer_detection.ipynb   <- notebook with outputs already baked in
└── README.md
```

## Setup & run (VSCode)

```bash
pip install tensorflow matplotlib numpy scipy scikit-learn seaborn opendatasets
```

**One-time Kaggle credential setup** (required — the notebook downloads
data directly from Kaggle):
1. Create a free Kaggle account if you don't have one.
2. Go to Kaggle → your profile → **Account** → **Create New API Token**.
   This downloads a `kaggle.json` file containing your username and key.
3. The first time you run the notebook's download cell, `opendatasets`
   will prompt for your Kaggle username and API key in the terminal —
   paste in the values from `kaggle.json` (or place `kaggle.json` in
   `~/.kaggle/` to skip the prompt).

Then open `cancer_detection.ipynb` in VSCode, select a Python kernel with
the packages above installed. Outputs are already saved in the notebook
from the original run; Run All to reproduce (downloads ~150MB of MRI
images on first run, then trains — expect a while on CPU).
