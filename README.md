# KNN Iris — Task 6

This repository contains the solution for **Task 6: K-Nearest Neighbors (KNN) Classification** using the **Iris.csv** dataset provided in the archive file.

## What's Included
- **`knn_task6.py`** — Main Python script that:
  - Loads the dataset (`Iris.csv`)
  - Encodes labels (`Species`) into numeric form
  - Normalizes features using `StandardScaler`
  - Trains KNN models for multiple `k` values
  - Reports accuracy and confusion matrices
  - Visualizes decision boundaries for petal features
- **`Iris.csv`** — Dataset extracted from the provided zip file.
- **`knn_task_outputs/`** — Saved CSV results:
  - `knn_accuracies.csv` — k values vs accuracy
  - `confusion_matrix_best_k.csv` — confusion matrix for the best k
- **`selected_dataset_preview.csv`** — First 100 rows preview of the dataset.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
