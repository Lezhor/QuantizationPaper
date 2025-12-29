# Efficient Post-Training Quantization for Compact CNNs

This repository contains the official implementation and experimental data for the paper **"Effiziente Post-Training Quantisierung kompakter CNNs: Naive Ansätze vs. Mixed Precision"**.

It provides a comprehensive evaluation of different Post-Training Quantization (PTQ) strategies on the CIFAR-10 dataset, focusing on the trade-off between inference latency, model size, and accuracy on CPU targets.

## 📌 Project Overview

Deploying Deep Learning models on edge devices requires efficient execution. This project explores how different quantization techniques affect a custom compact CNN architecture.

**Key Techniques Implemented:**
1.  **Baseline Training:** FP32 training of a custom CNN on CIFAR-10.
2.  **Naive PTQ:** Standard static quantization (INT8) using PyTorch's `QuantStub`.
3.  **Bias Correction:** Statistical correction of layer bias to mitigate quantization error.
4.  **Batch Normalization (BN) Folding:** Fusing Conv+BN+ReLU layers before quantization to reduce memory access and latency.
5.  **Mixed Precision Quantization (MPQ):** Sensitivity-based approach (using MSE) to keep sensitive layers in FP32.

## 📂 Repository Structure

```text
.
├── notebooks/
│   ├── export/                       # Generated diagrams
│   ├── data/                         # This is where the CIFAR10 Dataset will be downloaded to
│   ├── models/                       # saved models
│   └── Quantization Training.ipynb   # Main experiment pipeline (Training, Quantization, Evaluation)
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## 🚀 Getting Started

The code is implemented in Python 3.12 using PyTorch.

### Usage
Open the Jupyter Notebook to reproduce the experiments:

```bash
jupyter notebook notebooks/"Quantization Training.ipynb"
```

The notebook performs the following steps sequentially:
1.  **Train** the FP32 baseline model (or load a pretrained one).
2.  **Calibrate** the model using the MinMaxObserver.
3.  **Apply** various quantization strategies (Naive, Bias Corr, Folding, MPQ).
4.  **Evaluate** accuracy, model size, and CPU inference latency.
5.  **Visualize** results (Pareto Frontier, Layer-wise MSE, etc.).

## 📊 Key Results

Our experiments show that **PTQ with BN Folding** offers the best trade-off for compact CNNs on CIFAR-10, outperforming complex Mixed Precision approaches due to lower overhead.

| Strategy | Accuracy (Top-1) | Latency (CPU) | Model Size | Speedup |
|----------|------------------|---------------|------------|---------|
| **FP32 Baseline** | 71.91% | 0.141 ms | 2.09 MB | 1.0x |
| **Naive PTQ** | 71.42% | 0.048 ms | 0.53 MB | 2.9x |
| **Bias Correction** | 71.42% | 0.047 ms | 0.53 MB | 3.0x |
| **BN Folding** | **71.49%** | **0.041 ms** | **0.53 MB** | **3.4x** |
| **Mixed Precision** | 71.46% | 0.067 ms | 0.58 MB | 2.1x |

*> Note: Latency measured on CPU with Batch Size = 64.*

### Visualizations
The notebook generates several plots to analyze the performance:
* **Pareto Frontier:** Visualization of the Accuracy-Latency trade-off.
* **Layer-wise Sensitivity:** MSE analysis to identify sensitive layers (basis for MPQ).
* **Confusion Matrices:** Detailed error analysis per class.

## 📜 Citation

If you use this code for your research, please cite the accompanying paper:

```bibtex
@misc{aragon2025quantization,
  author = {Aragon, Daniil},
  title = {Effiziente Post-Training Quantisierung kompakter CNNs: Naive Ansätze vs. Mixed Precision},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Lezhor/QuantizationPaper}}
}
```

## 📄 License
This project is licensed under the MIT License.
