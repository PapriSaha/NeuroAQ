# NeuroAQ: A Hierarchical Multi-Task Residual Transformer for Quantifying Public Health Risks from Air Pollution

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

**NeuroAQ** introduces a **Hierarchical Multi-Task Residual Transformer (H-MTRT)** that simultaneously predicts continuous PM₂.₅ concentrations and classifies health risk levels in a single unified architecture. Unlike prior approaches that treat regression and classification independently, NeuroAQ jointly optimizes both tasks through a novel **Difficulty-Weighted Hybrid Objective** combining Focal Loss with tanh-modulated regression penalties.

### Key Features

- **Multi-Task Architecture** — Shared residual transformer backbone with dual heads for PM₂.₅ regression and 3-class health risk classification
- **Difficulty-Weighted Hybrid Loss** — Adaptive regression loss (tanh-weighted MSE) + Focal Loss for class imbalance
- **Uncertainty Quantification** — Monte Carlo Dropout-based Bayesian inference with 95% confidence intervals
- **Explainability** — Integrated Gradients attribution for feature importance analysis
- **Real-Time Health Alerts** — Color-coded public health alert system based on PM₂.₅ thresholds

### Results

| Metric | Value |
|--------|:-----:|
| R² Score | **0.9985** |
| RMSE | **1.79 µg/m³** |
| MAE | **0.94 µg/m³** |
| Pearson Correlation | **0.9993** |
| Classification Accuracy | **97.27%** |
| Average AUC (ROC) | **0.9986** |

## Repository Structure

```
NeuroAQ/
├── Neuro_AQ.ipynb          # Complete training & evaluation notebook
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data/
│   └── README.md           # Dataset download instructions
├── figures/                # Generated figures
│   ├── Diagram_NeuroAQ.png
│   ├── Fig_1a_LossCurve.png
│   ├── Fig_1b_BayesianPrediction.png
│   ├── Fig_1c_ConfusionMatrix.png
│   ├── Fig_1d_ROC.png
│   ├── Fig_2a_PrecisionRecall.png
│   ├── Fig_2b_PearsonCorrelation.png
│   ├── Fig_2c_ErrorDistribution.png
│   ├── Fig_3a_FeatureImportance.png
│   ├── Fig_3b_FeatureImpact.png
│   ├── Fig_3c_FeatureInteraction.png
│   ├── Fig_4_HealthAlerts.png
│   └── R2-plot.png
└── models/                 # Saved model checkpoints
    └── best_neuroaq_model.pt
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/PapriSaha/NeuroAQ.git
cd NeuroAQ
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the [Air Quality from Multiple Districts of Bangladesh](https://www.kaggle.com/datasets/mdshobuj/air-quality-dataset-of-bangladesh) dataset from Kaggle and place the CSV file in the `data/` directory.

### 4. Run the Notebook

Open `Neuro_AQ.ipynb` in **Google Colab** or **Jupyter Notebook** and run all cells. The notebook will:

1. Ingest and preprocess the dataset (temporal features, state persistence, robust scaling)
2. Train the H-MTRT model for 150 epochs with AdamW optimizer
3. Generate all figures
4. Save the best model checkpoint

> **Note:** The notebook is configured for Google Colab with Google Drive paths. Update `RAW_SOURCE_PATH` and `save_dir` if running locally.

## Model Architecture

```
Input (26 features)
    │
    ├── Linear(26 → 512) + GELU
    │
    ├── NeuroResidualBlock × 2
    │   └── Linear → LayerNorm → GELU → Linear → LayerNorm + Skip Connection
    │
    ├── Dropout(0.1)  ← MC Dropout for uncertainty
    │
    ├── Regression Head → Linear(512 → 1)  → PM₂.₅ prediction
    └── Classification Head → Linear(512 → 3) → Health risk class
```

### Training Configuration

| Parameter | Value |
|-----------|:-----:|
| Latent Dimension (d_model) | 512 |
| Residual Blocks | 2 |
| Activation | GELU |
| Optimizer | AdamW (lr=5×10⁻⁴, wd=10⁻⁵) |
| Epochs | 150 |
| Batch Size | 32 |
| Focal Loss γ | 2.0 |
| Task Balance λ | 0.6 |
| Gradient Clipping | max_norm=1.0 |
| MC Dropout Samples | 50 |
| Random Seed | 42 |

### Loss Function

The **Difficulty-Weighted Hybrid Objective** jointly optimizes regression and classification:

$$\mathcal{J} = \mathcal{L}_{reg} + \lambda \mathcal{L}_{cls}$$

where:
**Adaptive Regression Loss:**
$\mathcal{L}_{reg} = \frac{1}{N} \sum_{i=1}^{N} \omega_i \cdot (y_i - \hat{y}_i)^2$ with $\omega_i = 1 + \tanh(|y_i - \hat{y}_i|)$

**Focal Loss:**
$$ \mathcal{L}_{cls} = -\alpha(1-p_t)^\gamma \log(p_t) $$

## Dataset

| Split | Samples | Percentage |
|-------|:-------:|:----------:|
| Total (after filtering) | 2,564 | 100% |
| Training | 2,179 | 85% |
| Testing | 256 | 10% |
| Calibration | 129 | 5% |
| Input Features | 26 | — |

**Source:** [Air Quality from Multiple Districts of Bangladesh](https://www.kaggle.com/datasets/mdshobuj/air-quality-dataset-of-bangladesh) (Kashem et al., 2025)

Features include: sensor readings (PM₁₀, PM₁.₀, MQ7, MQ135), meteorological variables (temperature, humidity, dew point), engineered temporal features (hour, day-of-week, weekend flag), and derived state features (State Persistence).

## Figures

| Figure | Description |
|--------|-------------|
| Fig 2a | Training vs. Validation Loss Convergence |
| Fig 2b | Bayesian Prediction with 95% Confidence Intervals |
| Fig 2c | Confusion Matrix (Accuracy = 97.27%) |
| Fig 2d | Multi-Class ROC Curves (Avg AUC = 0.9986) |
| Fig 3a | Precision-Recall Curves |
| Fig 3b | Pearson Correlation Heatmap |
| Fig 3c | Error Distribution (μ=0.43, σ=1.74) |
| Fig 4a | Top 10 Feature Drivers (Integrated Gradients) |
| Fig 4b | Feature Impact Distribution |
| Fig 4c | Feature Interaction Correlation Matrix |
| Fig 5a | Public Health Alert Timeline |
| Fig 5b | Professional R² Plot with Density |

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{neuroaq2026,
  title     = {NeuroAQ: A Hierarchical Multi-Task Residual Transformer for Quantifying Public Health Risks from Air Pollution},
  author    = {Papri Saha, Ahmed Mostakim Fahim, Fairuz Zahin Sneha, Marmita Paul, Mohammad Rabiul Islam},
  booktitle = {Proceedings of the 4th International Conference on Computing Advancements (ICCA 2026)},
  year      = {2026},
  publisher = {ACM},
  doi       = {}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Air Quality from Multiple Districts of Bangladesh](https://www.kaggle.com/datasets/mdshobuj/air-quality-dataset-of-bangladesh) by Kashem et al.
- Conference: [ICCA 2026](https://icca.aiub.edu) — 4th International Conference on Computing Advancements, AIUB, Dhaka
