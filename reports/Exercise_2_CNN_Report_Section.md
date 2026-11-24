# Exercise 2: Convolutional Neural Network (CNN) Classifier (~630 Words)

*Author: Pooja (Member 2 — CNN Baseline & Visual Standards)*

---

## 2.1 Baseline CNN Architecture and Experimental Configuration

To investigate the impact of spatial convolution on natural image classification, the three-layer Multi-Layer Perceptron (`CIFAR10_MLP`) from Exercise 1 was replaced with a baseline Convolutional Neural Network (`CIFAR10_CNN`) containing **strictly three trainable layers** (two 2D convolutional layers followed by one fully connected classification head), interleaved with non-trainable Rectified Linear Unit (`ReLU`) activations and $2 \times 2$ max-pooling (`MaxPool2d`) operations:

1. **Trainable Layer 1 (`Conv2d`)**: Accepts the normalised $3 \times 32 \times 32$ RGB tensor and applies $32$ kernels of size $3 \times 3$ (`stride = 1`, `padding = 1`), preserving spatial resolution at $32 \times 32 \times 32$ (`896` trainable parameters: $32 \times 3 \times 3 \times 3 + 32$). A `ReLU` activation introduces non-linearity, followed by `MaxPool2d(kernel_size = 2, stride = 2)` to downsample feature maps to $32 \times 16 \times 16$.
2. **Trainable Layer 2 (`Conv2d`)**: Applies $64$ kernels of size $3 \times 3$ (`stride = 1`, `padding = 1`) across the $32$ input channels (`18,496` trainable parameters: $64 \times 32 \times 3 \times 3 + 64$), followed by `ReLU` and `MaxPool2d(2, 2)`, yielding a $64 \times 8 \times 8$ representation.
3. **Trainable Layer 3 (`Linear`)**: Flattens the $64 \times 8 \times 8$ feature tensor into a $4,096$-dimensional vector and maps it directly to the $10$ CIFAR-10 output class logits (`40,970` trainable parameters: $4,096 \times 10 + 10$).

In total, `CIFAR10_CNN` contains only **`60,362` trainable parameters**—a **$28.3\times$ reduction ($96.46\%$ fewer parameters)** compared to the `1,707,274` parameters of `CIFAR10_MLP`. To ensure a strictly controlled comparison, all dataset partitions and training hyperparameters were held identical to Exercise 1: the $50,000$ CIFAR-10 training images were split using `SEED = 42` into $45,000$ training and $5,000$ validation samples (keeping the $10,000$-image test partition completely separate), and the model was trained for $20$ epochs with a batch size of $64$, `CrossEntropyLoss`, and the `Adam` optimiser ($\eta = 0.001$).

---

## 2.2 Comparative Results: 3-Layer MLP vs. 3-Layer CNN

Table 2.1 and Figures 2.1–2.2 summarise the quantitative comparison between the Exercise 1 MLP and Exercise 2 CNN across all evaluation metrics.

**Table 2.1: Controlled comparison between Exercise 1 (`CIFAR10_MLP`) and Exercise 2 (`CIFAR10_CNN`).**

| Performance / Complexity Metric | Exercise 1: 3-Layer MLP | Exercise 2: 3-Layer CNN | Difference (CNN vs. MLP) |
| :--- | :---: | :---: | :---: |
| **Trainable Layers** | 3 (`3 × Linear`) | 3 (`2 × Conv2d + 1 × Linear`) | Spatial Conv Feature Extractor |
| **Total Trainable Parameters** | `1,707,274` | `60,362` | `-1,646,912` (**28.3× fewer**) |
| **Final Training Loss (Epoch 20)** | `0.5091` | `0.2701` | `-0.2390` |
| **Lowest Validation Loss** | `1.4030` (Epoch 6) | `0.8618` (Epoch 5) | `-0.5412` |
| **Final Validation Loss (Epoch 20)** | `2.2960` | `1.3069` | `-0.9891` |
| **Final Training Accuracy (Epoch 20)** | `81.79%` | `90.40%` | `+8.61%` |
| **Best Validation Accuracy** | `52.60%` (Epoch 8) | `71.70%` (Epoch 7) | **`+19.10%`** |
| **Unseen Test Accuracy (10,000 imgs)** | `52.53%` | `71.20%` | **`+18.67%`** |
| **Macro F1-Score (Test Partition)** | `0.5211` | `0.7099` | **`+0.1888`** |
| **Generalisation Gap (Train − Test Acc)** | `29.26%` | `19.20%` | **`-10.06%`** |

*(Embed `notebooks/ex2_cnn_curves.png` as **Figure 2.1** and `notebooks/ex2_mlp_vs_cnn_comparison.png` as **Figure 2.2** in the compiled PDF report).*

Already in Epoch 1, the CNN reached $62.76\%$ validation accuracy—surpassing the MLP's 20-epoch peak of $52.60\%$. The CNN achieved a peak validation accuracy of **$71.70\%$** (Epoch 7) and an unseen test accuracy of **$71.20\%$** ($+18.67$ percentage points above the MLP). Per-class F1-scores improved substantially across both rigid vehicle categories (`automobile`: $0.8224$ vs. $0.6494$; `truck`: $0.8022$ vs. $0.5701$; `ship`: $0.7974$ vs. $0.6575$) and deformable animal classes (`cat`: $0.5350$ vs. $0.3505$; `bird`: $0.6069$ vs. $0.4009$; `deer`: $0.6597$ vs. $0.3954$).

---

## 2.3 Discussion: How Convolutional Operations Influence Image Feature Learning

The superior generalisation and parameter efficiency of `CIFAR10_CNN` stem from four inductive biases inherent to convolutional architectures:

1. **Preservation of 2D Spatial Topology via Local Receptive Fields**: The MLP flattens each $3 \times 32 \times 32$ image into a 1D vector of $3,072$ scalars prior to the first layer, discarding 2D neighbourhood relationships and forcing the network to treat adjacent pixels identically to distant corner pixels. Conversely, the CNN's $3 \times 3$ convolutional kernels operate directly on local 2D patches, capturing spatially correlated patterns such as oriented edges, colour gradients, and corners.
2. **Weight Sharing and Parameter Efficiency**: In the MLP, the first dense layer (`Linear(3072 -> 512)`) consumes $1,573,376$ parameters because every hidden neuron learns a separate weight for every pixel coordinate. In `CIFAR10_CNN`, each $3 \times 3$ filter slides across all spatial locations using the same shared weights, enabling the two convolutional layers to extract rich visual representations using only $19,392$ parameters combined ($896 + 18,496$). This massive reduction in degrees of freedom acts as a structural regulariser that narrows the train–test generalisation gap from $29.26\%$ to $19.20\%$.
3. **Spatial Downsampling and Translation Invariance**: The $2 \times 2$ `MaxPool2d` operations retain the strongest local activations while halving spatial dimensions ($32 \times 32 \rightarrow 16 \times 16 \rightarrow 8 \times 8$). This expands the effective receptive field of subsequent layers and provides robustness against small spatial shifts or object translations within the frame.
4. **Hierarchical Feature Composition and Residual Overfitting**: The first convolutional layer (`32` filters) learns low-level primitives (edges and colour contrasts), whereas the second convolutional layer (`64` filters) combines those feature maps into higher-level semantic motifs (e.g., wheels, wings, animal silhouettes). Nevertheless, as shown in Figure 2.1, once training accuracy climbs past $\sim 80\%$ (after Epoch 7), validation loss gradually rises from $0.8618$ to $1.3069$ while training accuracy reaches $90.40\%$. This residual overfitting primarily arises in the $40,970$-parameter dense classification head (`Linear(4096 -> 10)`) memorising training activations, directly motivating the Dropout and Data Augmentation regularisation strategies evaluated in **Exercise 3**.

---

## Appendix: Baseline CNN Handoff Notes for Pair 2 (Bihansa — Ex 3 & Desandu — Ex 4)

- **Baseline Model Class**: `CIFAR10_CNN` in `notebooks/Exercise_2_CNN.ipynb`
- **Baseline Metrics to Compare Against**:
  - **Trainable Parameters**: `60,362`
  - **Final Train Loss / Train Acc (Epoch 20)**: `0.2701` / `90.40%`
  - **Lowest Val Loss / Best Val Acc**: `0.8618` (Epoch 5) / `71.70%` (Epoch 7)
  - **Final Test Accuracy (10,000 images)**: `71.20%`
  - **Generalisation Gap (Train Acc − Test Acc)**: `19.20%`
- **Shared Plotting Module**: Import `plot_learning_curves` from `src/visualization.py` so Exercise 3 and Exercise 4 charts match Exercise 1 and Exercise 2 exactly.
