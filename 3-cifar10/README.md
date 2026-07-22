# CIFAR-10 Image Classification

## Objective

Classify 32×32 color images into 10 categories (airplane, automobile,
bird, cat, deer, dog, frog, horse, ship, truck) using a Convolutional
Neural Network.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), loaded via
`tensorflow.keras.datasets.cifar10.load_data()`. **No manual download
needed** — Keras downloads and caches it automatically on first run
(60,000 images total: 50,000 train / 10,000 test).

## Libraries Used

- `tensorflow` / `keras` — CNN model, training callbacks, data augmentation
- `scikit-learn` — classification metrics
- `matplotlib`, `seaborn` — visualizations, confusion matrix
- `numpy`

## Methodology

1. Loaded CIFAR-10 (50,000 train / 10,000 test images), inspected shapes
   and sample images per class.
2. Normalized pixel values to the [0, 1] range.
3. Applied data augmentation (rotation, width/height shift, horizontal
   flip) to the training set.
4. Built a CNN (multiple convolutional blocks with BatchNorm + Dropout)
   and trained for 30 epochs with `ReduceLROnPlateau`.
5. Evaluated on the held-out test set with accuracy/loss curves, sample
   predictions, a full classification report, and a confusion matrix.

## Results

**Test Accuracy: 87.16%** (Test Loss: 0.3974)
Best validation accuracy: 87.16%, with a training/validation gap of only
~-1% — no significant overfitting.

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Airplane | 0.88 | 0.91 | 0.89 |
| Automobile | 0.91 | 0.96 | 0.93 |
| Bird | 0.88 | 0.80 | 0.84 |
| Cat | 0.86 | 0.66 | 0.75 |
| Deer | 0.84 | 0.88 | 0.86 |
| Dog | 0.88 | 0.75 | 0.81 |
| Frog | 0.80 | 0.96 | 0.87 |
| Horse | 0.88 | 0.92 | 0.90 |
| Ship | 0.93 | 0.93 | 0.93 |
| Truck | 0.88 | 0.95 | 0.91 |
| **Weighted avg** | **0.87** | **0.87** | **0.87** |

## Conclusion

The CNN reaches 87.2% test accuracy on CIFAR-10, close to the ~88% target
noted in the original run, with no meaningful overfitting thanks to data
augmentation and batch normalization. Ship and automobile are recognized
most reliably (F1 ≈ 0.93), while **cat** is the weakest class by far
(recall 0.66 — the model misses over a third of actual cats, likely
confusing them with dogs or other four-legged animals). This is a common
CIFAR-10 pattern: at 32×32 resolution, visually similar animal categories
are genuinely harder for a CNN to separate than more distinct classes like
vehicles.

---

## Project structure

```
3-cifar10/
├── cifar10.ipynb   <- notebook with outputs already baked in
└── README.md
```

## Setup & run (VSCode)

```bash
pip install tensorflow matplotlib numpy scipy scikit-learn seaborn
```

Open `cifar10.ipynb` in VSCode, select a Python kernel with those packages
installed. The dataset downloads automatically via Keras on first run —
no manual download or path fixing needed. Outputs are already saved in
the notebook from the original run; Run All to reproduce (expect ~30-60
min on CPU, faster with a GPU).
