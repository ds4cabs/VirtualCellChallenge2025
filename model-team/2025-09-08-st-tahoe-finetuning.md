# ST-Tahoe Fine-Tuning: A Case Study in Representation and Strategy

### TL;DR
A series of fine-tuning experiments on the ST-Tahoe model revealed critical lessons. Initial rounds with embedding-based inputs failed due to a representation mismatch with the evaluation metric. Switching the input to raw gene expression, to match the evaluator, dramatically improved the score from **<0.2 to ~5.7**. Subsequently, unfreezing the model's backbone and continuing to train pushed the score to **~8.5**. Key takeaways are: align your training and evaluation data spaces, analyze specific metrics over single scores, and use a "freeze-then-unfreeze" strategy for efficient fine-tuning.

---

### Background & Goal
This post documents a series of fine‑tuning attempts on **ST‑Tahoe**, the State Transition module of the Arc Institute's "State" model. The goal was to improve the **cell‑eval** score—computed in the **18,080‑dim** gene‑expression space—by fine-tuning the model, which was originally trained on small-molecule perturbation data.

### The Experimental Journey: A Summary

- **Round 1 — 2025-08-26**
  - **Configuration:** `tahoe-finetune-0826_config.yaml`
  - **Inputs:** The model used 2,058-dim SE600M embeddings for the cell state (`X_state`) and 5,120-dim ESM2 embeddings for the perturbations (from `protein_embeddings.pt`).
  - **Model:** The `PertSets` model was initialized from a pre-trained ST-Tahoe checkpoint. The loss function was a weighted sum of the embedding loss and the gene decoder loss (`total_loss = loss_emb + weight * decoder_loss`), with the `decoder_weight` set to `0.5`.
  - **Backbone:** The transformer backbone was kept frozen (`freeze_pert_backbone: true`).
  - **Training:** The model was trained for 8,000 steps with a learning rate of `0.0003` and a batch size of `32`.
  - **Inference:** The 4,000-step checkpoint was used for inference.
  - **Outcome:** The `cell-eval` score was extremely low (0.19), with a very high mean absolute error (0.1909). This suggested that the model was not effectively predicting the full gene expression space, possibly due to the low weight on the decoder loss.

- **Round 2 — 2025-09-01**
  - **Configuration:** `tahoe-finetune-0901_config.yaml`
  - **Inputs:** Same as Round 1.
  - **Model:** Initialized from the best checkpoint of Round 1. To address the poor scores from the first round, the `decoder_weight` in the loss function was increased to `1.0`, putting more emphasis on the gene expression prediction task.
  - **Backbone:** The transformer backbone remained frozen.
  - **Training:** Trained for 4,000 steps with a learning rate of `5.0e-05` and a batch size of `48`.
  - **Inference:** The final 4,000-step checkpoint was used for inference.
  - **Outcome:** This change successfully validated the hypothesis that increasing the decoder weight would improve the gene expression prediction. The mean absolute error improved dramatically from 0.1909 to **0.0303**. However, the overall score dropped to 0.06, indicating that this change, while directionally correct for the MSE, was not enough to overcome the larger representation mismatch problem and negatively impacted other components of the score.

- **Round 3: Switching to Raw Expression (Backbone Frozen)**
  - **Date:** 2025-09-05 & 2025-09-06
  - **Configurations:** `tahoe-finetune-0905_config.yaml`, `tahoe-finetune-0906_config.yaml`
  - **Key Change:** This round marked the most significant shift in strategy: the model input was switched from embeddings to the **full 18,080-dimensional raw gene expression** to align the model's training with the evaluation metric.
  - **Model & Training:** The experiment began with a short, 1,000-step run (`0905`) with the backbone kept **frozen**. This initial test showed immediate promise: the validation loss (now directly measuring gene expression error, equivalent to the `val/decoder_loss` in previous rounds) dropped to **~11**, a significant improvement over the ~15 seen previously. This strong signal justified continuing the experiment with a longer run (`0906`).
  - **Outcome:** A massive improvement. By training on the same representation that the evaluator uses, the final `cell-eval` score jumped from less than 0.2 to **~5.7**. This confirmed the primary hypothesis that the representation mismatch was the main cause of the poor performance.

- **Round 4: Unfreezing the Backbone**
  - **Date:** 2025-09-07 & 2025-09-08
  - **Configurations:** `tahoe-finetune-0907_config.yaml`, `tahoe-finetune-0908_config.yaml`
  - **Key Change:** Based on the success of Round 3, the transformer backbone was **unfrozen** to allow the entire model to fine-tune on the raw expression data.
  - **Model:** These experiments were initialized from the *last* checkpoint of the Round 3 (`0906`) experiment, not the checkpoint with the lowest validation loss. This was a suboptimal choice that highlights the importance of careful checkpoint selection.
  - **Training:** The model was trained for a much longer duration (a combined ~50,000 steps across the `0907` and `0908` runs) with a low learning rate. A subtle and persistent issue was discovered during this phase. The initial `0907` run (10,000 steps) showed periodic fluctuations in the training loss. The cause was traced to the learning rate scheduler state being carried over from the Round 3 checkpoint. An attempt was made to fix this in the `0908` run by ensuring no `last.ckpt` was present, but the issue persisted, suggesting the scheduler state was saved in the main checkpoint file itself. This highlights a critical pitfall: when resuming training, one must be certain that all optimizer and scheduler states are either correctly re-initialized or intentionally loaded.
  - **Outcome:** Despite the scheduler issue, unfreezing the backbone led to a further significant improvement. The best validation score was found at step 18,000 of the extended run. Using this checkpoint for inference pushed the `cell-eval` score to its highest point yet: **~8.5**. This demonstrates that after the crucial input-space alignment, further performance can be gained by fine-tuning the entire model, even when not starting from the optimal checkpoint and with a partially-behaving learning rate scheduler.

### Key Lessons Learned

1.  **Representation Mismatch is a Critical Failure Point**: The most important lesson is that the model's training data representation must align with the evaluation metric's representation. Training on 2,058-dimensional embeddings while evaluating on 18,080-dimensional raw expression was the primary reason for the failure of the first two rounds.

2.  **Targeted Interventions Can Be Validated with Specific Metrics**: Simply looking at the overall score would have hidden the success of the change made in Round 2. Increasing the `decoder_weight` *did* improve the Mean Absolute Error, as intended. This shows the importance of analyzing all relevant metrics to understand the full impact of a change.

3.  **Checkpointing Strategy is Not an Afterthought**: The suboptimal choice to start Round 4 from the *last* checkpoint instead of the *best* one from Round 3 could have limited the final performance. The best model checkpoint is a valuable asset and should be used strategically.

4.  **Optimizer and Scheduler States Persist in Checkpoints**: The learning rate scheduler issues in Round 4 highlight a critical pitfall: when resuming training, optimizer and scheduler states can be loaded along with model weights, leading to unexpected behavior. One must be certain that all optimizer and scheduler states are either correctly re-initialized or intentionally loaded.

### Practical Recommendations

1. **Representation & objective alignment**
   - If the evaluator is in 18,080 dims, optimize there.
   - If you still want to leverage SE600M: use it as an **auxiliary signal** (e.g., **distillation** or **contrastive alignment**) rather than a hard input replacement.
2. **Checkpoint policy**
   - Treat **min-val** as the canonical inference ckpt; keep fixed-step ckpts only for ablations.
3. **Freeze → Unfreeze schedule**
   - Start frozen to stabilize heads and reduce cost.
   - Unfreeze with **warmup + cosine/one‑cycle**; add **weight decay, dropout, and grad clipping**.
4. **Engineering hygiene**
   - Enforce a stable **ckpt naming convention** (e.g., `best-val.pt`).
   - The use of a tool like **Weights & Biases (`wandb`)** was critical for logging, plotting, and comparing runs, which helped detect divergence early.

### Reproducibility
- All runs can be reconstructed from the corresponding `***_config.yaml` files.
- Validation minima and training horizons are taken from `***_metrics.csv` files.
