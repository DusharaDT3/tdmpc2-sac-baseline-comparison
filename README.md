# TD-MPC2 Phase 2 — SAC baseline vs. pretrained checkpoint comparison (Dushara, KIC-BSCCSAI-251P-052)

This is the implementation referenced by the Phase 2 report submitted on the LMS. It's a small,
self-contained package of this project's own work. It does **not** include the official `tdmpc2`
repository, the downloaded pretrained checkpoints, or the Python virtual environment — those are large,
externally-available dependencies, obtained as described below.

## What's here

- `Phase2_TD_MPC2_Dushara_052_baseline.ipynb` — the Phase 2 deliverable notebook, already executed with
  real outputs embedded. Open it directly to see the full workflow, explanations, tables, and chart
  without re-running anything.
- `sac_baseline.py` — the script that trained the SAC baseline and produced `sac_baseline_results.csv`.
- `stress_test_noise.py` — the observation-noise robustness script (Phase 3 bonus material, not a Phase 2
  requirement).
- `tier1_pretrained_results.csv`, `sac_baseline_results.csv`, `stress_test_noise_results.csv` — the small
  result files the notebook loads by default.
- `training_evidence/` — genuine evidence that real training/evaluation runs happened:
  - `sweep_eval_results.csv` — final evaluation returns for all 5 runs of the Phase 3 bonus
    training-steps/model-size sweep (real `train.py` runs, read from their own `eval.csv` files).
  - `sample_training_log_excerpt.txt` — a real excerpt (first/last 15 lines) of one sweep run's actual
    training log, showing the real per-episode training-reward progression.
  - `overrides_size5_steps50k.yaml`, `overrides_tier1_cheetah_run_eval.yaml` — the exact command-line
    overrides Hydra recorded for two real runs, copied verbatim (not retyped from memory).
- `requirements.txt` — the verified package versions this project's own code depends on.

## Setup

1. Clone the official repository and check out the exact revision used for this work:
   ```bash
   git clone https://github.com/nicklashansen/tdmpc2.git
   cd tdmpc2 && git checkout e9f59321933cbc8e11a002b842adc7d4ffae8ff1
   ```
2. Download the 3 pretrained single-task checkpoints into `tdmpc2/checkpoints/`:
   `cheetah-run-1.pt`, `walker-walk-1.pt`, `hopper-hop-1.pt`, from
   `huggingface.co/nicklashansen/tdmpc2/tree/main/dmcontrol`.
3. Create a fresh Python virtual environment and install the versions listed in `requirements.txt`. Do
   not install anything else into the same environment afterwards without re-checking `numpy`'s version —
   several unrelated packages (`pandas`, `matplotlib`) have been observed to silently pull in an
   incompatible `numpy 2.x` as a transitive dependency, which breaks `dm_control`'s `env.reset()`.
4. Place this project's folder structure so that, relative to wherever the notebook is opened,
   `tdmpc2/tdmpc2/` (the repo's own inner `tdmpc2` package directory) and `tdmpc2/checkpoints/` (the
   downloaded weights) both resolve correctly — matching the layout the notebook's `TDMPC2_DIR`/`CKPT_DIR`
   variables expect.

## How to run

- **Just view the results (no setup beyond this folder needed):** open the notebook. Every result cell
  already has its real output embedded.
- **Regenerate the table/chart from the saved CSVs:** run the notebook top-to-bottom with `RUN_LIVE` and
  `RUN_LIVE_SAC` left at `False` (the default). No GPU or checkpoints required; completes in well under a
  minute.
- **Re-run the Tier 1 (pretrained TD-MPC2) evaluation genuinely:** set `RUN_LIVE=True`. Needs the official
  repo, the 3 downloaded checkpoints, and a CUDA GPU (the codebase hardcodes `cuda:0`, no CPU fallback).
  Takes roughly 1-2 minutes per task.
- **Retrain SAC genuinely:** set `RUN_LIVE_SAC=True`. Works on CPU or GPU. Takes roughly 20 minutes per
  task (~1 hour for all 3). No model is saved by this code — this matches how the original results were
  produced.
- **The Phase 3 bonus training-steps/model-size sweep is not reproduced from inside the notebook.** Its
  exact commands are recorded in `training_evidence/overrides_size5_steps50k.yaml` (one example); running
  the same `train.py` command for each of the 5 configurations listed in the notebook's Section 5.2 would
  reproduce it. Budget roughly 1-3 hours per run (unattended, GPU required).

## Which steps are expensive / need a GPU

| Step | GPU needed? | Approx. time |
|---|---|---|
| View saved results (default notebook run) | No | <1 minute |
| Re-run Tier 1 evaluation (`RUN_LIVE=True`) | Yes (hardcoded `cuda:0`) | 1-2 min/task |
| Retrain SAC (`RUN_LIVE_SAC=True`) | No (CPU works, slower) | ~20 min/task |
| Re-run the bonus sweep | Yes | 1-3 hours/run |
