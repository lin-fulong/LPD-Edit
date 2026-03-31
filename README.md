<div align="center">
  <h1>LPD-Edit</h1>
  <p><strong>Latent Principal Denoising for Robust Lifelong Knowledge Editing</strong></p>
  <p align="center">
    <a href="#">📄 Paper</a> •
    <a href="https://github.com/lin-fulong/LPD-Edit">🚀 GitHub</a>
  </p>
</div>

LPD-Edit is a robust lifelong knowledge editing method designed for long-horizon sequential editing.  
It improves editing stability by introducing latent principal denoising and anchored weighted fusion prior to parameter-shift estimation.

> Key idea: denoise hidden-state editing signals in a principal semantic subspace and fuse them with the original representations to obtain stable yet plastic parameter updates.

## Overview

Lifelong knowledge editing aims to continuously update factual knowledge in large language models without full retraining. In long-horizon sequential editing, cached hidden-state representations may accumulate semantic noise and spurious directions, which can degrade parameter-shift estimation and destabilize later edits.

LPD-Edit addresses this issue through two coupled components. First, a PCA-based denoising module projects hidden states into a principal subspace to preserve dominant semantic directions while suppressing noisy variation. Second, an anchored weighted fusion module combines denoised features with the original hidden states to balance model plasticity and stability and reduce deviation from the pre-training distribution.

The method is implemented in a unified experimental framework with support for multiple datasets, model backbones, and editor configurations.

## Data & Model Preparation

We use publicly available datasets and model resources in our experiments.

1️⃣ Download the files from [Google Drive](https://drive.google.com/drive/folders/1wsxG5Ybf6hT9QUlccvzTuJSfL_TFNyKQ?usp=sharing) and place them under `data/raw`.

2️⃣ Download the [UltraEditBench](https://huggingface.co/datasets/XiaojieGu/UltraEditBench) and save it under `data/raw/ultraeditbench`.

3️⃣ Specify the path to model weights by setting the `name_or_path` field in the selected file under `config/model/`.

If you need to use locate-then-edit methods, publicly available precomputed covariance matrices for several models can be found on Hugging Face: [GPT-J 6B](https://huggingface.co/XiaojieGu/gpt-j-6b_CovarianceMatrix), [Qwen2.5-7B-Instruct](https://huggingface.co/XiaojieGu/Qwen2.5-7B-Instruct_CovarianceMatrix), [Mistral-7B-v0.3](https://huggingface.co/XiaojieGu/Mistral-7B-v0.3_CovarianceMatrix), [LLaMA-3-8B-Instruct](https://huggingface.co/XiaojieGu/Llama-3-8B-Instruct_CovarianceMatrix). 

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

## Running Experiments

We follow standard experimental setups from prior work to ensure fair comparison and reproducibility.

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

- `batch_size`: number of edit samples processed in parallel during each editing step
- `n_edits`: number of edit requests applied per step in sequential editing
- `num_seq`: total number of sequential editing steps (i.e., length of the editing horizon)
- `pca_denoise.enable_pca`: whether PCA denoising is enabled
- `pca_denoise.var_threshold`: cumulative explained-variance threshold for selecting principal components
- `alpha`: interpolation factor between original and PCA-denoised hidden states

## Acknowledgements

This work is built on top of the UltraEdit framework and also benefits from prior open-source research efforts on model editing. We sincerely acknowledge the contributions of the corresponding authors and maintainers to the model editing community.

<!-- ## Citation

If you find this work helpful, please consider citing:

```bibtex
@misc{liulin2026lpdedit,
  title={LPD-Edit: Latent Principal Denoising for Robust Lifelong Knowledge Editing},
  author={Ruinan Liu and Fulong Lin},
  year={2026},
  note={Under review}
} -->
