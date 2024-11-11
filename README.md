# FED-HARGPT: A Federated Transformer-based Architecture for Human Activity Recognition

This repository hosts the code for paper ''**FED-HARGPT**: A Hybrid Centralized-Federated Approach of a Transformer-based Architecture for Human Context Recognition''. The project leverages federated learning (FL) with the Flower framework to balance data privacy and model performance in scenarios with non-IID data.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Contributions](#contributions)
- [Citation](#citation)

## Overview

The increasing ubiquity of mobile sensors (such as smartphones and wearables) has facilitated discreet monitoring of human activities. This project explores a federated learning model, derived from a centralized baseline, that is effective in non-IID conditions, thus allowing for robust, privacy-preserving HAR. The **FED-HARGPT** model is based on the GPT-2 architecture, modified for multi-label classification of human activities using mobile sensor data from the ExtraSensory dataset.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FED-HARGPT.git
   cd FED-HARGPT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. [Optional] To use GPU for model training, ensure that PyTorch with CUDA support is installed.

## Usage

### Centralized Training

To train the model on a centralized dataset, use the `main.py` script. This process fine-tunes the model's initial weights, which are used in federated training. The following example runs centralized training with a default configuration:

```bash
python main.py
```

### Federated Training

1. Start the federated server by running:
   ```bash
   python server.py
   ```

2. Each client device should run `client_har_fed.py` with the required data path and user identifier. For example:

   ```bash
   python client_har_fed.py --exp <experiment_number> --user <user_id>
   ```

### Model Hyperparameter Tuning

Hyperparameter tuning can be done using `main.py` and Optuna. The script uses a cross-validation approach, and configurations are provided in the `config` dictionary in `main.py`.

```bash
python main.py --optuna
```

## Methodology

The **FED-HARGPT** model adapts GPT-2's architecture for the multi-label classification task of HAR. Key methodological steps include:

1. **Centralized Training**: The model is first trained centrally using 48 subjects, establishing a robust baseline and initializing weights.
2. **Federated Learning with Flower Framework**: The model is then distributed to clients (each representing a subset of users), where it is further fine-tuned locally and aggregated by the federated server.
3. **Performance Metrics**: The model’s performance is evaluated primarily using Balanced Accuracy (BA) to account for class imbalance in the HAR dataset.

For more details, refer to the [paper](paper).

## Contributions

We welcome contributions to improve the model, code structure, or add new features. Please submit a pull request with a brief description of your changes.


## Citation
