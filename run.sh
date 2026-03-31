python main.py dataset=zsre model=llama-3-instruct editor=lpdedit num_seq=200 \
    editor.cache_dir=cache \
    dataset.batch_size=10 \
    dataset.n_edits=100 \
    model.edit_modules="[model.layers.11.mlp.gate_proj, model.layers.12.mlp.gate_proj, model.layers.13.mlp.gate_proj, model.layers.14.mlp.gate_proj, model.layers.15.mlp.gate_proj, model.layers.18.mlp.up_proj, model.layers.19.mlp.up_proj, model.layers.20.mlp.up_proj, model.layers.21.mlp.up_proj, model.layers.22.mlp.up_proj, model.layers.23.mlp.up_proj, model.layers.24.mlp.up_proj]" \
    editor.pca_denoise.enable_pca=true \
    editor.pca_denoise.var_threshold=0.85 \
    editor.alpha=0.3
