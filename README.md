<h1 align="center">LPD-Edit</h1>

<p align="center">
  <strong>PCA-Enhanced Lifelong Model Editing</strong>
</p>

<p align="center">
  <a href="#">📄 Paper</a> •
  <a href="#">🤗 Dataset</a> •
  <a href="https://arxiv.org/abs/2505.14679">UltraEdit</a>
</p>

<p align="center">
  <em>Official research code for LPD-Edit, a PCA-enhanced lifelong model editing method built on top of UltraEdit.</em>
</p>

LPD-Edit is a lifelong model editing method built on top of the UltraEdit framework.  
It improves sequential editing stability by introducing PCA-based denoising and representation fusion before parameter-shift estimation.

> Key idea: denoise hidden-state editing signals in a low-rank principal subspace to obtain more stable and reusable parameter updates during long-horizon sequential editing.

## Overview

Long-horizon sequential editing can accumulate noise and spurious directions in cached hidden-state representations. These noisy editing signals may degrade the quality of parameter-shift estimation and lead to unstable behavior as the number of edits grows.

LPD-Edit addresses this issue by projecting editing representations into a principal subspace and fusing the denoised signal with the original hidden states. This design aims to preserve dominant semantic directions while suppressing noisy variation, without changing the overall editing pipeline inherited from UltraEdit.

LPD-Edit is implemented in a unified experimental framework with support for multiple datasets, model backbones, and editor configurations.

## Highlights

- PCA-enhanced sequential model editing within the UltraEdit parameter-shift framework
- Denoising of cached hidden-state representations before parameter-shift prediction
- Representation fusion that balances stability and information preservation
- Plug-and-play integration into the existing UltraEdit-style editing pipeline
- Unified Hydra-based configuration for datasets, models, and editor settings

## Method

LPD-Edit follows the parameter-shift prediction framework of UltraEdit. Given cached hidden-state representations collected during editing, it first performs PCA-based subspace projection to suppress noisy directions. The denoised representations are then fused with the original hidden states:

```text
H' = (1 - alpha) * H + alpha * H_k
```

where `H` is the original hidden-state representation, `H_k` is the PCA reconstruction from the selected principal components, and `alpha` controls the denoising strength.

The fused representation `H'` is then used in the downstream parameter-shift computation. This design improves the quality and stability of editing features while preserving the efficiency and overall workflow of the original UltraEdit pipeline.

## Results Snapshot

LPD-Edit is designed for long-horizon sequential editing, where stability becomes increasingly important as the number of edits grows. We recommend evaluating it jointly with edit success, preservation, and long-horizon stability metrics.

## Quick Start

### 1. Create the environment

```bash
conda create -n lpdedit python=3.10
conda activate lpdedit
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Adjust the PyTorch installation command if your CUDA version or hardware environment is different.

### 2. Prepare the model path

Edit one of the model configuration files under `config/model/` and set the local checkpoint path through `name_or_path`, for example:

- `config/model/llama-3-instruct.yaml`
- `config/model/mistral-7b.yaml`
- `config/model/qwen2.5-7b.yaml`

### 3. Run a minimal test

```bash
python main.py dataset=zsre model=llama-3-instruct editor=lpdedit num_seq=2 \
    editor.cache_dir=cache_test \
    dataset.batch_size=1 \
    dataset.n_edits=2 \
    editor.pca_denoise.enable_pca=true \
    editor.pca_denoise.var_threshold=0.85 \
    editor.alpha=0.3
```

This command is intended as a sanity check rather than a final experiment. It helps confirm the following:

- the model checkpoint can be loaded correctly
- the editing pipeline can execute at least a short run
- cache files can be written to disk
- the PCA denoising module is connected correctly

## Repository Structure

```text
LPD-Edit/
├── config/                   # Hydra configuration files
├── data/                     # Dataset loaders and raw data
├── editor/                   # Editing algorithms and PCA denoising utilities
├── main.py                   # Experiment entry point
├── model.py                  # Model loading logic
├── util.py                   # Shared utility functions
├── run.sh                    # Example command for running experiments
└── README.md
```

### Key Files

- `editor/lpdedit.py`: main implementation of LPD-Edit
- `editor/pca_denoise/project.py`: PCA-based denoising and reconstruction
- `editor/pca_denoise/select_k.py`: principal component selection
- `config/editor/lpdedit.yaml`: default LPD-Edit hyperparameters
- `main.py`: experiment entry point

Compared with UltraEdit, the main modification is introduced in `predict_param_shifts`, where PCA-based denoising and representation fusion are applied before downstream parameter-shift estimation.

## Data Preparation

Place dataset files under `data/raw/` according to the paths specified in `config/dataset/`.

Supported datasets include:

- `zsre`
- `fever`
- `wikibigedit`
- `ultraeditbench`

Example raw data directories:

- `data/raw/zsre/`
- `data/raw/fever/`
- `data/raw/wikibigedit/`
- `data/raw/ultraeditbench/`

Make sure the local directory structure matches the corresponding dataset configuration files.

## Model Preparation

Set the local checkpoint path through the `name_or_path` field in the selected file under `config/model/`, such as:

- `config/model/llama-3-instruct.yaml`
- `config/model/mistral-7b.yaml`
- `config/model/qwen2.5-7b.yaml`

If you use locate-then-edit style methods in the broader experimental framework, precomputed covariance matrices may also be needed for some model backbones. In that case, prepare the corresponding covariance files separately and place them in the expected local path before running experiments.

## Data & Model Preparation

1. Download the files from Google Drive and place them under `data/raw/`.
2. Download `UltraEditBench` and place it under `data/raw/ultraeditbench/`.
3. Specify the path to model weights by setting the `name_or_path` field in the selected file under `config/model/`.

If you need to use locate-then-edit methods, precomputed covariance matrices may be required for several model backbones, including GPT-J 6B, Qwen2.5-7B-Instruct, Mistral-7B-v0.3, LLaMA-3-8B-Instruct, and LLaMA-2-7B-hf.

## Running Experiments

The project uses Hydra for experiment configuration management. The default entry point is:

```bash
python main.py
```

An example command is provided in `run.sh`:

```bash
python main.py dataset=zsre model=llama-3-instruct editor=lpdedit num_seq=200 \
    editor.cache_dir=cache \
    dataset.batch_size=10 \
    dataset.n_edits=100 \
    model.edit_modules="[model.layers.11.mlp.gate_proj, model.layers.12.mlp.gate_proj, model.layers.13.mlp.gate_proj, model.layers.14.mlp.gate_proj, model.layers.15.mlp.gate_proj, model.layers.18.mlp.up_proj, model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj]" \
    editor.pca_denoise.enable_pca=true \
    editor.pca_denoise.var_threshold=0.85 \
    editor.alpha=0.3
```

Alternatively:

```bash
sh run.sh
```

## Important Hyperparameters

The default LPD-Edit configuration is defined in `config/editor/lpdedit.yaml`. Key hyperparameters include:

- `lr`: scaling factor used in parameter-shift estimation
- `batch_size`: internal batch size used by the editor
- `cache_dir`: directory for storing cached keys and value gradients
- `alpha`: interpolation factor between original and PCA-denoised hidden states
- `pca_denoise.enable_pca`: whether PCA denoising is enabled
- `pca_denoise.var_threshold`: cumulative explained-variance threshold for selecting principal components
- `pca_denoise.min_k`: minimum number of retained principal components
- `pca_denoise.eps`: numerical stability term

## Supported Editors

The repository currently contains multiple editor implementations, including:

- `UltraEdit`
- `LPD-Edit`
- `RLEdit`
- `MEND`
- `MALMEN`

LPD-Edit is the main method introduced in this repository.

## Logging

The current implementation initializes experiment logging with `wandb` in `main.py`. By default, the run is organized as:

- `project = {dataset.name}_{model.name}`
- `name = {editor.name}_{dataset.n_edits}`

Users who do not wish to use Weights & Biases may modify or disable the corresponding initialization logic in `main.py`.

## Reproducibility

For reproducible results, please report the following:

- model backbone and exact checkpoint version
- dataset and split configuration
- editing layers specified in `model.edit_modules`
- PCA hyperparameters such as `alpha`, `var_threshold`, and `min_k`
- number of sequential edits such as `num_seq` and `dataset.n_edits`
- random seed
- PyTorch version
- CUDA version
- GPU type and memory

Logs and caches are organized according to the configured `cache_dir` and logging backend.

## Notes and Limitations

- LPD-Edit is currently implemented within an UltraEdit-style parameter-shift framework
- performance and memory usage depend on the chosen backbone, edited layers, and cache size
- PCA denoising may require tuning of `alpha` and `var_threshold` across different models and datasets
- some shared editor utilities are inherited from a common codebase, so extending the framework to new editors may require additional adaptation

## Relationship to UltraEdit

This repository is not a direct mirror of the original UltraEdit project. Instead, it is an extended research codebase built on top of the UltraEdit editing framework. The main contribution of this repository is the implementation of LPD-Edit, which augments the original pipeline with PCA-based denoising and representation fusion during parameter-shift prediction.

Accordingly, this repository should be understood as a research implementation of **LPD-Edit based on UltraEdit**, rather than as the original UltraEdit release.

## Acknowledgements

This work is built on top of the UltraEdit framework and also benefits from prior open-source research efforts on model editing. We sincerely acknowledge the contributions of the corresponding authors and maintainers to the model editing community.

## Citation

If you find this repository useful in your research, please cite the corresponding LPD-Edit paper. If your manuscript is still in preparation, you may temporarily use a placeholder entry such as:

```bibtex
@misc{lpdedit2026,
  title={LPD-Edit: PCA-Enhanced Lifelong Model Editing},
  author={Author1 and Author2 and Author3},
  year={2026},
  note={Manuscript in preparation}
}
```
