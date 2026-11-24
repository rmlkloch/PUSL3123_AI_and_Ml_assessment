# PUSL3123 — AI and Machine Learning (Assessment 1: Set Exercises)

PyTorch implementation and experimental analysis of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures, regularisation techniques, and model scaling on the **CIFAR-10** image classification dataset.

---

## Repository Structure

```text
PUSL3123_AI_and_Ml_assessment/
├── notebooks/
│   ├── Exercise_1_MLP.ipynb              # Exercise 1: 3-Layer MLP Baseline (Lochana)
│   ├── ex1_mlp_curves.png                # Exercise 1: MLP Training & Validation Curves
│   ├── Exercise_2_CNN.ipynb              # Exercise 2: 3-Layer CNN Baseline & Comparison (Pooja)
│   ├── ex2_cnn_curves.png                # Exercise 2: CNN Training & Validation Curves
│   └── ex2_mlp_vs_cnn_comparison.png     # Exercise 2: MLP vs. CNN Comparative Analysis Charts
├── reports/
│   └── Exercise_2_CNN_Report_Section.md  # Exercise 2 (~600-word) Report Section & Handoff Notes (Pooja)
├── src/
│   ├── data_loader.py                    # Shared CIFAR-10 PyTorch DataLoaders (45k Train / 5k Val / 10k Test)
│   ├── dataset_setup.py                  # Dataset verification script
│   └── visualization.py                  # Unified 300-DPI Matplotlib plotting & comparison table utilities
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup & Reproducibility

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Dataset Partitioning (`src/data_loader.py`)**:
   - Uses `torchvision.datasets.CIFAR10` (`60,000` RGB images of size `32 × 32` across `10` classes).
   - Standard channel-wise normalisation: `mean = (0.4914, 0.4822, 0.4465)`, `std = (0.2470, 0.2435, 0.2616)`.
   - Reproducible training/validation split (`SEED = 42`, `val_split = 0.1`):
     - **Training Set**: `45,000` images (`704` batches at `batch_size = 64`)
     - **Validation Set**: `5,000` images (`79` batches at `batch_size = 64`)
     - **Test Set**: `10,000` images (`157` batches at `batch_size = 64`, kept strictly separate from training and model selection)

---

## Baseline Architectures & Shared Visual Utilities

### Exercise 1: 3-Layer Multi-Layer Perceptron (`CIFAR10_MLP`)
- **Architecture**: `Flatten(3072) -> Linear(3072, 512) -> ReLU -> Linear(512, 256) -> ReLU -> Linear(256, 10)`
- **Trainable Parameters**: `1,707,274`
- **Training Configuration**: `20` epochs, `CrossEntropyLoss`, `Adam(lr=0.001)`, `batch_size=64`, `SEED=42`

### Exercise 2: 3-Layer Convolutional Neural Network (`CIFAR10_CNN`)
- **Architecture** (3 trainable layers: 2 convolutional + 1 fully connected):
  1. `Conv2d(3, 32, kernel_size=3, padding=1)` + `ReLU()` + `MaxPool2d(2, 2)` $\rightarrow (32 \times 16 \times 16)$
  2. `Conv2d(32, 64, kernel_size=3, padding=1)` + `ReLU()` + `MaxPool2d(2, 2)` $\rightarrow (64 \times 8 \times 8)$
  3. `Flatten()` + `Linear(4096, 10)` $\rightarrow (10)$
- **Trainable Parameters**: `60,362` (`28.3×` fewer parameters than `CIFAR10_MLP`)
- **Training Configuration**: Identical to Exercise 1 (`20` epochs, `CrossEntropyLoss`, `Adam(lr=0.001)`, `batch_size=64`, `SEED=42`)

### Shared Visual Analytics (`src/visualization.py`)
All notebooks can import the unified plotting functions from `src/visualization.py` to keep report figures visually consistent:
- `plot_learning_curves(...)`: Side-by-side 300-DPI Training vs. Validation Loss and Accuracy curves.
- `plot_mlp_vs_cnn_comparison(...)`: 2×2 comparative dashboard (loss trajectories, accuracy trajectories, split accuracy bars, and trainable parameter counts).
- `build_comparison_table(...)`: Formatted Pandas DataFrame comparing model performance and complexity.
