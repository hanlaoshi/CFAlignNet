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

<p align="center">
  <img src="figures/architecture.png" alt="CFAlignNet Architecture" width="100%">
</p>

## 📁 Repository Structure

```
CFAlignNet/
├── data_provider/
│   ├── data_factory.py          # Dataset factory and data loading
│   └── data_loader.py           # Dataset classes for all benchmarks
├── dataset/
│   └── citynetwork.csv          # Network Traffic Dataset
│   └── ETTh1.csv                # Electricity Transformer Temperature Dataset 1
│   └── ETTh2.csv                # Electricity Transformer Temperature Dataset 2
│   └── M5.csv                   # Retail Sales Dataset
│   └── ML_IOT.csv               # Traffic Flow Dataset
│   └── residensial_data.csv     # Residential Load Dataset
├── layers/
│   ├── Embed.py                 # Patch embedding layer
│   └── StandardNorm.py          # Reversible instance normalization
├── models/
│   └── CFAlignNet.py            # Main model implementation
├── utils/
│   ├── metrics.py               # Evaluation metrics (MAE, MSE)
│   └── tools.py                 # Training utilities and visualization
├── scripts/
│   ├── CFAlignNet.sh            # City Network dataset
│   ├── etth1.sh                 # ETTh1 dataset
│   ├── etth2.sh                 # ETTh2 dataset
│   ├── iot.sh                   # IoT dataset
│   ├── m5.sh                    # M5 dataset
│   └── residential_data.sh      # Residential Load dataset
├── run.py                       # Main entry point for training and evaluation
├── requirements.txt             # Python dependencies
└── README.md
```

## 🛠️ Requirements

The code is built with PyTorch. You can install the dependencies with the following command:

```bash
conda create -n cfalignnet python=3.8
conda activate cfalignnet
pip install -r requirements.txt
```

Key dependencies:

- Python 3.8+
- PyTorch >= 1.10
- Transformers (HuggingFace)
- PEFT (Parameter-Efficient Fine-Tuning)
- Accelerate (HuggingFace)
- Einops

## 📂 Datasets

We evaluate the model on the following 6 real-world datasets. Please place your data files in the `dataset/` directory.

| Dataset | Domain | Variables | Frequency | Prediction Horizons |
| --- | --- | --- | --- | --- |
| **City Network** | Network Traffic | 3 | Hourly | 192, 720，1440, 2640 |
| **ETTh1** | Energy (Transformer Temp.) | 7 | Hourly | 192, 720， 1440, 2640|
| **ETTh2** | Energy (Transformer Temp.) | 7 | Hourly | 192, 720，1440, 2640 |
| **Residential** | Energy (Residential Load) | 3 | Hourly | 192, 720，1440, 2640 |
| **IoT** | Traffic Flow (IoT Sensor) | 3 | Hourly | 192, 720， 1440, 2640|
| **M5** | Retail Sales | 3 | Daily | 8, 30，60，110 |

## 🚀 Usage

### 1. Training

To train CFAlignNet on a specific dataset, use the corresponding script in the `scripts/` directory:

```bash
# City Network Traffic dataset
bash scripts/CFAlignNet.sh

# ETTh1 dataset
bash scripts/etth1.sh

# ETTh2 dataset
bash scripts/etth2.sh

# Residential Load dataset
bash scripts/residential_data.sh

# IoT dataset
bash scripts/iot.sh

# M5 dataset
bash scripts/m5.sh
```

Alternatively, you can run training directly with custom parameters:

```bash
accelerate launch --multi_gpu --num_processes 2 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path city_network.csv \
  --model_id CFAlignNet \
  --model CFAlignNet \
  --data City_Network \
  --features M \
  --seq_len 512 \
  --label_len 256 \
  --pred_len 1440 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 16 \
  --d_model 16 \
  --d_ff 32 \
  --patch_len 16 \
  --batch_size 2 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
```

### 2. Evaluation

To evaluate a pre-trained model:

```bash
accelerate launch --multi_gpu --num_processes 2 run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path city_network.csv \
  --model_id CFAlignNet \
  --model CFAlignNet \
  --data City_Network \
  --features M \
  --seq_len 512 \
  --pred_len 1440 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --d_model 16 \
  --d_ff 32
```

## 📊 Main Results

### Network Traffic Dataset (Long-term Forecasting)

| Methods | Horizon | MSE | MAE |
| --- | --- | --- | --- |
| **CFAlignNet** | **2640** | **0.669** | **0.616** |
| TimeCMA | 2640 | 1.204 | 0.808 |
| DFGCN | 2640 | 1.073 | 0.704 |
| TimeLLM | 2640 | 1.301 | 0.788 |

*For full results across all 6 datasets, please refer to Table 2 in our paper.*

## 📈 Visualization

CFAlignNet provides built-in visualization tools for analyzing model predictions, particularly during holiday periods where distribution shifts are most pronounced.

### Holiday Prediction Visualization

The framework generates two types of visualizations for each holiday event:

1. **Full Prediction Window:** Shows the complete prediction alongside ground truth, with the holiday period highlighted, and overlays central trend curves (rolling average) to illustrate the model's ability to capture overall trajectory shifts.

2. **Magnified Holiday View:** A zoomed-in plot focusing specifically on the holiday period, enabling detailed comparison between predicted and actual values during critical transition points.

To generate holiday visualizations after training:

```python
from utils.tools import visualize_holiday_predictions

visualize_holiday_predictions(
    args, model, test_data, test_loader,
    accelerator, criterion, mae_metric, mape_metric
)
```

Visualization outputs are saved to `holiday_predictions/` directory as PDF files.

### Gating Weight Analysis

The dual-branch alignment mechanism employs a learnable gating function to dynamically balance OT-aligned and attention-aligned features. The gate weight $g$ (Eq. 10 in the paper) can be monitored during inference to analyze the relative contribution of each alignment branch across different forecasting scenarios.

## 📜 Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{han5650682cfalignnet,
  title={CFAlignNet: Knowledge-Enhanced Coarse-Fine Cross-modal Alignment for Long-term Network Traffic Forecasting},
  author={Han, Aifu and Ye, Zifeng and Zhou, Yang and Huang, Xiaoxia},
  journal={Available at SSRN 5650682}
}
```

## 🙏 Acknowledgments

This work was supported by the National Natural Science Foundation of China (U22A2003, 62271515). We appreciate the following open-source works that inspired our code: [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [Time-LLM](https://github.com/KimMeen/Time-LLM).
