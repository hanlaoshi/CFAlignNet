# CFAlignNet: Knowledge-Enhanced Coarse-Fine Cross-modal Alignment for Long-term Network Traffic Forecasting

This is the official PyTorch implementation of the paper: **"CFAlignNet: Knowledge-Enhanced Coarse-Fine Cross-modal Alignment for Long-term Network Traffic Forecasting"**.

> **Abstract:** Long-term network traffic forecasting faces significant challenges from distribution shifts caused by sudden events (e.g., holidays). We propose **CFAlignNet**, a framework that integrates **Large Language Models (LLMs)** with a **coarse-fine dual-branch cross-modal alignment** mechanism. By encoding historical statistics and event timestamps into structured prompts, and employing Optimal Transport (coarse) alongside Cross-Attention (fine), CFAlignNet effectively bridges the modality gap and captures both global distribution consistency and local semantic dependencies.

## 🌟 Key Features

- **Knowledge-Enhanced Prompting:** Encodes temporal characteristics and event timestamps into natural language, enabling the model to implicitly capture event dynamics (e.g., holidays, extreme weather) from sparse historical occurrences.
- **Dual-Branch Alignment:**
   - **Coarse-Grained:** Utilizes **Optimal Transport (OT)** to enforce global distribution consistency between time series and text embeddings.
   - **Fine-Grained:** Employs **Cross-Attention** for precise local feature correspondence.

- **SOTA Performance:** Achieves state-of-the-art results on 6 real-world datasets, with a **37.7% MSE reduction** on the China Mobile network traffic dataset compared to previous baselines.

## 🏗️ Model Architecture

*(Note: Please ensure the image path matches your repository structure)*

## 🛠️ Requirements

The code is built with PyTorch. You can install the dependencies with the following command:

```
conda create -n cfalignnet python=3.8
conda activate cfalignnet
pip install -r requirements.txt

```

Key dependencies:

- Python 3.8+
- PyTorch >= 1.10
- Transformers (HuggingFace)
- POT (Python Optimal Transport)
- Einops

## 📂 Datasets

We evaluate the model on the following datasets. Please refer to the `data/` directory for preprocessing scripts.

- **Network Traffic:** Real-world hourly traffic data from China Mobile (Cities A/B/C).
- **Energy:** ETTh1, ETTh2, and Residential Load (Guangxi Province).
- **Others:** Traffic Flow (IoT-sensor), Retail Sales (M5).

## 🚀 Usage

### 1. Training

To train CFAlignNet on the Network Traffic dataset with the default settings:

```
python run.py \
  --is_training 1 \
  --root_path ./dataset/network_traffic/ \
  --data_path traffic.csv \
  --model_id network_traffic_192_1440 \
  --model CFAlignNet \
  --data custom \
  --features M \
  --seq_len 192 \
  --pred_len 1440 \
  --enc_in 3 \
  --llm_model gpt2 \
  --llm_dim 768 \
  --patch_len 16 \
  --stride 8 

```

### 2. Evaluation

To evaluate a pre-trained model:

```
python run.py \
  --is_training 0 \
  --model_id network_traffic_192_1440 \
  --model CFAlignNet \
  --data custom \
  --pred_len 1440

```

## 📊 Main Results

Performance comparison on the **Network Traffic** dataset (Long-term forecasting):

| Methods | Horizon | MAE | MSE |
| --- | --- | --- | --- |
| **CFAlignNet** | **1440** | **0.593** | **0.641** |
| **CFAlignNet** | **2640** | **0.616** | **0.669** |
| TimeCMA | 2640 | 0.808 | 1.204 |
| DFGCN | 2640 | 0.704 | 1.073 |
| TimeLLM | 2640 | 0.788 | 1.301 |

*For full results across all datasets (ETTh1/2, M5, etc.), please refer to Table 2 in our paper.*

## 📜 Citation

If you find this repository useful for your research, please consider citing our paper:

```
@article{han5650682cfalignnet,
  title={CFAlignNet: Knowledge-Enhanced Coarse-Fine Cross-modal Alignment for Long-term Network Traffic Forecasting},
  author={Han, Aifu and Ye, Zifeng and Zhou, Yang and Huang, Xiaoxia},
  journal={Available at SSRN 5650682}
}

```

## 🙏 Acknowledgments

This work was supported by the National Natural Science Foundation of China (U22A2003, 62271515). We appreciate the following open-source works that inspired our code: [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [TimeLLM](https://github.com/KimMeen/Time-LLM).
