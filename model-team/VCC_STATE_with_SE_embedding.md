# 🚀 VCC Processing Workflow

## **Step 1 — Generate Embeddings for All Input Files**
```python
import os
os.environ['MPLBACKEND'] = 'Agg'

# 1. Files to process
FILENAMES_TO_PROCESS = [
    "competition_train.h5",
    "jurkat.h5",
    "k562.h5",
    "k562_gwps.h5",
    "rpe1.h5",
    "competition_val_template.h5ad",
    "hepg2.h5"
]

# 2. Model & directory configuration
model_folder = "Path/to/VCC/arcinstitute/SE-600M/"
checkpoint_path = "Path/to/VCC/arcinstitute/SE-600M/se600m_epoch16.ckpt"
input_dir = "./competition_support_set"
output_dir = "./vci_pretrain"

# 3. Create output directory
os.makedirs(output_dir, exist_ok=True)

# 4. Loop over files
for filename in FILENAMES_TO_PROCESS:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"✅ Skipping {filename}, already exists.")
        continue

    print(f"🚀 Processing {filename} ...")
    !state emb transform         --model-folder {model_folder}         --checkpoint {checkpoint_path}         --input {input_path}         --output {output_path}
    print(f"✅ Finished {filename}\n")
```

---

## **Step 2 — Train Model**
```bash
state tx train   data.kwargs.toml_config_path="vci_pretrain/starter.toml"   data.kwargs.embed_key="X_state"   data.kwargs.num_workers=4   data.kwargs.batch_col="batch_var"   data.kwargs.pert_col="target_gene"   data.kwargs.cell_type_key="cell_type"   data.kwargs.control_pert="non-targeting"   data.kwargs.perturbation_features_file="vci_pretrain/ESM2_pert_features.pt"   training.max_steps=8000   training.ckpt_every_n_steps=1000   model=tahoe_sm   wandb.tags="[first_run]"   output_dir="competition_state"   name="first_run"
```

---

## **Step 3 — Run Inference on Validation Data**
```bash
state tx infer   --embed_key X_state   --output "competition_se/prediction.h5ad"   --model_dir "competition_se/first_run"   --checkpoint "competition_se/first_run/checkpoints/last.ckpt"   --adata "vci_pretrain/competition_val_template.h5ad"   --pert_col "target_gene"
```

---

## **Leaderboard Result**
### That's it! You can now upload the vcc file to the leaderboard. We hope contestants will improve significantly on this baseline.For reference, after 8000 steps of training, this model generated the following unnormalized scores:

DE Score: 0.183

MAE Score: 0.334

Perturbation Score: 0.579

And the following normalized overall score: 0.072
