# ESc-GNN: Equivariant Geometric Scattering-based Geometric Neural Networks

## 1&nbsp;&nbsp;Project overview
ESc-GNN is a PyTorch/PyTorch-Geometric framework for building rotationally-equivariant graph models that utilize diffusion wavelet-based geometric scattering on both scalar and vector node features.

The repository contains:
* the ESc-GNN model implementation (`models/escgnn_modular.py`)
* implementations for comparison models (some adapted from [Geometric GNN Dojo](https://github.com/chaitjo/geometric-gnn-dojo))
* data-processing utilities for computing ESc-GNN's scalar- and vector-track sparse diffusion operators `P, Q` and storing them in an HDF5 file (`data_processing/`)
* reusable configuration dataclasses (`config/`)
* an (optionally) DDP training pipeline that uses **🤗 Accelerate** for multi-GPU training, `wandb` for experiment tracking, and includes example `slurm` scripts
* a script for generating the synthetic ellipsoidal geometric graph data sets used in our experiments (`scripts/python/generate_ellipsoid_dataset.py`)
* a script for recreating the figures illustrating the SO(d)-equivariance of our vector diffusion wavelets (`theory_verification/vector_diffusion_equivariance_viz.py`)

---
## 2&nbsp;&nbsp;Required packages
Core dependencies:
- python>=3.11
- pytorch
- torch-geometric
- torchmetrics
- accelerate
- numpy
- scikit-learn
- h5py
- pyyaml

Optional dependencies:
- torch-scatter[1]
- torch-cluster[1]
- torch-sparse[1]
- e3nn (for Tensor Field Networks models, etc.)
- wandb
- pandas
- matplotlib

[1] Can be installed as a dependency of pytorch-geometric. See the [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---
## 3&nbsp;&nbsp;Training pipeline example
Below is a minimal end-to-end recipe for training an ESc-GNN model.

### 3.1 Create (or copy) a YAML config
Adapt `config/yaml_files/<model_name>.yaml` files as needed.

>Note that for multi-model experiments, you may create a subdirectory within the `config/yaml_files/` directory, an `experiment.yaml` config, and then individual model yaml config files. In this setup, the individual model parameters will override the experiment-wide parameters, allowing minimal duplication of parameter specification across config files. (Note that any parameters not specified in the experiment or model yaml files will revert to the defaults in `model_config.py`, `dataset_config.py`, etc.). For $k$-fold cross-validation experiments, set `experiment_type: kfold` in the `experiment.yaml` config, and also set, e.g., `use_cv_test_set: true` and `k_folds: 5`.

### 3.2 Pre-compute diffusion operators
```bash
python scripts/python/parallel_process_dataset.py \
    --config_path config/yaml_files/{your_config}.yaml \
    --dataset <dataset>
```
This script builds the sparse diffusion operators **P** and **Q** in parallel
and saves them to the HDF5 file defined at `<dataset>.h5_path`.

### 3.3 Launch training
Single-GPU quick start:
```bash
python scripts/python/main_training.py \
  --config_path config/yaml_files/{your_config}.yaml \
  --dataset <dataset>
```

Multi-GPU / SLURM example (uses Distributed Data Parallel via 🤗 Accelerate; modify the SLURM job settings in the file first):
```bash
sbatch scripts/slurm/train.sh \
  --config={your_config}.yaml \
  --dataset=<dataset>
```
The script automatically
* creates an experiment folder structure
  `experiments/<dataset>/<model>/{models,logs,metrics}`
* supports resuming with
  `--snapshot_path /path/to/experiments/.../models/{best,checkpoint,final}`
* enables Weights & Biases logging (online or offline) if `use_wandb_logging: true` in the YAML.

---