# Face Recognition

## Objective

Classify face images by identity using a Convolutional Neural Network,
trained on the LFW (Labeled Faces in the Wild) "people" subset —
recognizing 7 individuals who each have at least 70 photos in the dataset.

## Dataset

[LFW People dataset](http://vis-www.cs.umass.edu/lfw/) via
`sklearn.datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)`.
**No manual download needed** — scikit-learn downloads and caches it
automatically on first run.

## Libraries Used

- `tensorflow` / `keras` — CNN model, training callbacks, data augmentation
- `scikit-learn` — dataset loading, train/test split, classification metrics
- `matplotlib`, `seaborn` — visualizations
- `numpy`, `scipy`

## Methodology

1. Loaded the LFW dataset (7 people, images resized to 50×37 grayscale).
2. Stratified 80/20 train/test split; computed class weights to correct
   for class imbalance (e.g. Bush: 530 images vs Chavez: 71 images).
3. Applied data augmentation (rotation, shifts, horizontal flip, zoom) to
   the training set.
4. Built a CNN (2 convolutional blocks with BatchNorm + Dropout, followed
   by dense layers) and trained with `ModelCheckpoint`,
   `ReduceLROnPlateau`, and `EarlyStopping` callbacks.
5. Evaluated the best checkpoint on the held-out test set with a full
   classification report and sample prediction visualizations.

## Results

**Test Accuracy: 97.29%**

| Person | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Sharon | 0.94 | 1.00 | 0.97 | 16 |
| Powell | 0.98 | 1.00 | 0.99 | 47 |
| Rumsfeld | 0.92 | 0.96 | 0.94 | 24 |
| Bush | 0.98 | 0.98 | 0.98 | 106 |
| Schroeder | 0.96 | 1.00 | 0.98 | 22 |
| Chavez | 1.00 | 1.00 | 1.00 | 14 |
| Blair | 1.00 | 0.86 | 0.93 | 29 |
| **Weighted avg** | **0.97** | **0.97** | **0.97** | **258** |

## Conclusion

The CNN achieves 97.3% test accuracy across all 7 individuals despite
significant class imbalance in the raw dataset (up to 7.5x more images for
the most-represented person). Class weighting and data augmentation were
both important for handling the imbalance without the model simply
defaulting to the majority class. The lowest recall (0.86, Blair) suggests
that person's face shares more visual similarity with others in the set
than the rest do with each other — a natural limitation when working with
a small number of classes and modest per-class sample sizes.

---

## Project structure

```
2-face-recognition/
├── face_recognition.ipynb   <- notebook with outputs already baked in
└── README.md
```

## Setup & run (VSCode)

```bash
pip install tensorflow matplotlib numpy scipy scikit-learn seaborn
```

Open `face_recognition.ipynb` in VSCode, select a Python kernel with those
packages installed. The dataset downloads automatically via scikit-learn
on first run — no manual download or path fixing needed. Outputs are
already saved in the notebook from the original run; Run All to reproduce.
