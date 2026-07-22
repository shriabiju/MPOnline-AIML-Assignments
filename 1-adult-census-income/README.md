# Adult Census Income

## Objective

Predict whether an individual's annual income exceeds $50K, based on
demographic and employment attributes from the 1994 US Census (the "Adult"
/ "Census Income" dataset). The notebook goes beyond a single model — it
builds and compares 5 classifiers, then improves the best ones by fixing
class imbalance and tuning hyperparameters.

## Dataset Link

[Adult / Census Income Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
(also mirrored on Kaggle as "Adult Census Income")

`adult.csv` is included in this folder — the dataset is public domain /
CC BY 4.0 licensed by UCI, so redistribution here is permitted.

## Libraries Used

- `pandas`, `numpy` — data loading and manipulation
- `matplotlib`, `seaborn` — visualizations (distributions, boxplots,
  correlation heatmap, confusion matrices, ROC curves)
- `scikit-learn` — `LabelEncoder`, `StandardScaler`, `train_test_split`,
  `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`,
  `KNeighborsClassifier`, `SVC`, and classification metrics

## Methodology

1. **Dataset Understanding** — loaded 32,561 rows × 15 columns, inspected
   dtypes, summary statistics, missing-value patterns, and target
   distribution (income is imbalanced: ~76% ≤50K vs ~24% >50K).
2. **Data Cleaning** — replaced the dataset's `'?'` placeholder with `NaN`,
   imputed missing categorical values with the mode, removed duplicate
   rows, stripped whitespace from string columns, and capped outliers in
   `capital.gain`, `capital.loss`, and `hours.per.week` using the IQR
   method.
3. **Feature Engineering** — encoded the target (`<=50K`→0, `>50K`→1),
   dropped `fnlwgt` (a sampling weight, not predictive) and `education`
   (redundant with `education.num`), label-encoded remaining categorical
   columns, examined a correlation heatmap, split 80/20 (stratified), and
   scaled features with `StandardScaler` for distance/gradient-based models.
4. **Model Building** — trained 5 classifiers: Logistic Regression,
   Decision Tree, Random Forest, KNN, and SVM.
5. **Performance Evaluation** — compared all 5 models on Accuracy,
   Precision, Recall, F1, and ROC-AUC, with confusion matrices and ROC
   curves for each.
6. **Model Finetuning & Improvement** — upsampled the minority class
   (>50K) to fix class imbalance, retrained 4 models (SVM excluded — too
   slow on the larger balanced set) with tuned hyperparameters, and
   compared F1 scores before vs. after.

## Results

**Best tuned model: KNN**

| Metric | Value |
|---|---|
| Accuracy | 0.8483 |
| Precision | 0.8058 |
| Recall | 0.9202 |
| F1 Score | 0.8592 |
| ROC-AUC | 0.9281 |

**F1 Score improvement from tuning + class balancing:**

| Algorithm | Before | After |
|---|---|---|
| Logistic Regression | 0.5205 | ~0.76 |
| Decision Tree | 0.6099 | ~0.83 |
| Random Forest | 0.5866 | ~0.84 |
| KNN | 0.5884 | **0.8592** |

Fixing the class imbalance (via upsampling) drove most of the improvement
— all four models' F1 scores rose substantially once the minority class
(>50K earners) was no longer underrepresented during training.

## Conclusion

Income is predictable from census attributes with reasonably high
accuracy — the tuned KNN model correctly classifies ~85% of individuals,
catching 92% of actual >50K earners (recall) with 81% precision. The
biggest single improvement didn't come from switching algorithms — it came
from addressing class imbalance: F1 scores across all four re-tuned models
roughly matched or exceeded their untuned Random Forest baseline once the
training data was rebalanced. A key limitation of the linear model here
(Logistic Regression) is that it assumes linear decision boundaries and
consistently underperformed the tree- and distance-based models, which can
capture non-linear relationships between features like age, education, and
hours worked. This suggests income is driven by non-linear interactions
between demographic factors rather than any single feature acting
independently.

---

## Project structure

```
1-adult-census-income/
├── adult.csv                          <- dataset (public domain, UCI)
├── adult_census_assignment.ipynb      <- notebook with all 6 tasks + outputs
└── README.md
```

## Setup & run (VSCode)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Open `adult_census_assignment.ipynb` in VSCode and select a Python kernel
with those packages installed. All outputs (tables, plots, metrics) are
already baked into the notebook from a full run against the real dataset
— "Run All" again only if you want to reproduce or modify it.
