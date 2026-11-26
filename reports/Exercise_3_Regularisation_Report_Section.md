# Exercise 3: Regularisation & Generalisation in CNNs (~750 Words)

*Author: Bihansa (Member 3 — Experimentation & Scaling Lead)*

---

## 3.1 Motivation, Experimental Design, and Hypotheses

In **Exercise 2**, the baseline Convolutional Neural Network (`CIFAR10_CNN`) demonstrated clear architectural superiority over the Multi-Layer Perceptron (`CIFAR10_MLP`), achieving a peak validation accuracy of $71.70\%$ and an unseen test accuracy of $71.20\%$ with $28.3\times$ fewer parameters ($60,362$ vs. $1,707,274$). However, detailed tracking of the baseline training trajectory reveals significant **overfitting**:
1. After Epoch 5, validation loss reaches its minimum at $0.8618$ and progressively degrades to $1.3069$ by Epoch 20 (a $+51.6\%$ divergence).
2. Training accuracy reaches $90.40\%$, producing a wide **generalisation gap of $19.20$ percentage points** against the test partition.
3. Because the convolutional feature extractor contains only $19,392$ parameters while the dense classification head contains $40,970$ parameters ($67.9\%$ of the entire model capacity), the dense head readily memorises sample-specific activation noise rather than invariant class boundaries.

To mitigate this overfitting, **Exercise 3** investigates two foundational regularisation techniques—**Dropout** and **Data Augmentation**—both independently and in combination under strictly identical training hyperparameters ($20$ epochs, Adam $\eta = 0.001$, batch size $64$, seed $42$, $45,000$ train / $5,000$ validation / $10,000$ test split):
- **Model A (Baseline CNN)**: Baseline 3-layer architecture without regularisation.
- **Model B (Dropout Only, $p = 0.5$)**: Dropout applied to the flattened $4,096$-dimensional feature vector preceding the linear classification layer.
- **Model C (Data Augmentation Only)**: Standard architecture trained with stochastic geometric transformations (`RandomCrop(32, padding=4, padding_mode="reflect")` and `RandomHorizontalFlip(p=0.5)`). Validation and test sets remain unaltered.
- **Model D (Dropout + Data Augmentation)**: Compound model integrating both latent-space stochasticity and input-space geometric invariance.

---

## 3.2 Quantitative Comparative Results

Table 3.1 and Figure 3.1 present the comparative performance and overfitting metrics across all four experimental configurations.

**Table 3.1: Comprehensive Regularisation & Generalisation Comparison on CIFAR-10.**

| Model Configuration | Total Parameters | Final Train Loss | Lowest Val Loss | Final Val Loss (Ep 20) | Final Train Acc | Best Val Acc | Unseen Test Acc | Generalisation Gap (Train − Test) | Improvement vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Baseline CNN** | $60,362$ | $0.2701$ | $0.8618$ (Ep 5) | $1.3069$ | $90.40\%$ | $71.70\%$ | $71.20\%$ | $19.20\%$ | Baseline |
| **Model B: Dropout ($p=0.5$)** | $60,362$ | $0.3885$ | $0.8175$ (Ep 8) | $0.9810$ | $86.45\%$ | $72.60\%$ | $72.35\%$ | $14.10\%$ | $+1.15\%$ |
| **Model C: Data Augmentation** | $60,362$ | $0.4750$ | $0.7060$ (Ep 13) | $0.7460$ | $83.70\%$ | $76.25\%$ | $75.85\%$ | $7.85\%$ | $+4.65\%$ |
| **Model D: Dropout + Data Aug** | $60,362$ | $0.5810$ | $0.6810$ (Ep 14) | $0.7040$ | $80.05\%$ | $77.10\%$ | **$76.70\%$** | **$3.35\%$** | **$+5.50\%$** |

*(Embed `notebooks/ex3_regularisation_curves.png` as **Figure 3.1** in the compiled PDF report).*

---

## 3.3 Critical Analysis: Regularisation Dynamics and Overfitting Mitigation

### 1. Independent Impact of Dropout ($p = 0.5$)
Introducing a dropout probability of $p = 0.5$ preceding the linear classification layer reduced the generalisation gap from $19.20\%$ to **$14.10\%$** (a $5.10$ percentage point drop in overfitting), while increasing final test accuracy from $71.20\%$ to **$72.35\%$** ($+1.15\%$). 
- *Mechanistic Principle*: Dropout operates as an implicit ensemble technique. During each training iteration, $50\%$ of the feature dimensions extracted by the second convolutional layer are randomly masked to zero. This disrupts co-adaptation between hidden units: no single neuron in the classification head can rely on the presence of a specific companion activation, forcing the network to learn redundant, highly distributed representations.
- *Loss Curve Behaviour*: While the baseline validation loss diverged sharply after Epoch 5, Dropout constrained the divergence, lowering the final validation loss from $1.3069$ to $0.9810$.

### 2. Independent Impact of Data Augmentation
Stochastic data augmentation exerted a substantially stronger regularisation effect than Dropout alone, slashing the generalisation gap from $19.20\%$ down to **$7.85\%$** (a **$59.1\%$ reduction in overfitting**) and boosting test accuracy to **$75.85\%$** ($+4.65\%$ over baseline).
- *Mechanistic Principle*: Convolutional networks are inherently translation-equivariant, but not naturally invariant to arbitrary spatial shifts or reflections. By introducing random horizontal flips and 4-pixel reflective crops, the training data distribution is artificially expanded over a continuous geometric manifold. Because the network rarely observes the exact same pixel grid across training epochs, memorising specific pixel coordinate arrangements becomes impossible.
- *Loss Curve Behaviour*: The validation loss trajectory remained stable through Epoch 13, bottoming out at $0.7060$ (compared to $0.8618$ in the baseline) and demonstrating sustained learning without premature saturation.

### 3. Compound Synergy: Combining Dropout and Data Augmentation
Combining both regularisation strategies (Model D) produced the optimal classifier, achieving **$76.70\%$ test accuracy** ($+5.50$ percentage points above baseline) and suppressing the generalisation gap to an exceptional **$3.35\%$**.
- *Orthogonal Regularisation Modes*: Data Augmentation acts externally on the **input manifold** by enforcing spatial invariance, whereas Dropout acts internally on the **latent representation space** by preventing parameter over-reliance.
- *Stability and Convergence*: The final validation loss of Model D ($0.7040$) closely mirrored its lowest validation loss ($0.6810$), completely arresting the runaway validation loss explosion observed in the unregularised baseline.

In conclusion, while Dropout effectively regularises high-capacity dense layers, Data Augmentation is the single most potent regulariser for vision tasks. Deploying both simultaneously produces an optimal inductive bias, delivering high test accuracy while almost entirely eliminating empirical overfitting.
