# Hotspot-Aware Evaluation of Learning-Based Routing-Congestion Prediction (CircuitNet-N14)

Code to reproduce the experiments in the paper *"Hotspot-Aware Evaluation of
Learning-Based Routing-Congestion Prediction on CircuitNet-N14"*.

This is an **evaluation / cautionary study**, not a new state-of-the-art method.
Its findings are reported honestly:

- A threshold-resolved **hotspot-region error** metric, reported together with
  **detection metrics** (precision, recall, F1, PR-AUC) and **cross-design** tests.
- Within a single design, a severity-weighted **hotspot loss** lowers severe-hotspot
  error (theta >= 0.6) by ~60% on held-out test data, but at a large cost to average
  fidelity, and **the gain comes from the objective, not model capacity**.
- **Detection metrics show the gain is over-prediction**: at theta = 0.6 the hotspot
  models reach recall ~0.83 but precision ~0.02 (they label almost every tile a
  hotspot).
- **Under cross-design evaluation the advantage disappears** (all models reach
  hotspot error ~0.9). The within-design gain is interpolation, not generalisation.
- Apparent gains against lightly trained baselines are partly an **undertraining
  artefact**.

Take-away: single-design evaluation substantially overstates the benefit of
specialised losses; a one-sided hotspot metric must be paired with detection
metrics and cross-design testing.

## Repository structure

```
src/model.py                     # the shared U-Net + Inception + self-attention backbone
                                 #   and the hotspot-weighted loss (HotspotLoss).
                                 #   In Colab this file is used as /content/deeproute_c_colab.py
scripts/
  test_split_experiment.py       # 70/15/15 train/val/TEST split, full 2x2 factorial,
                                 #   multi-seed, test-set metrics  (main within-design result)
  generalization_experiment.py   # train on one design, test on UNSEEN designs (cross-design)
  detection_metrics.py           # precision/recall/F1/PR-AUC + predicted-/non-hotspot error
  lambda_tau_sweep.py            # sweep of the loss weight (lambda) and threshold (tau)
  rebuild_table.py               # retrain all configs under one converged schedule
  resume_and_improve.py          # warm-start from a saved checkpoint and continue training
  revision_experiments.py        # threshold-sensitivity + multi-seed + leave-one-design-out
  save_all_to_drive.py           # archive checkpoints/results/figures to Google Drive
requirements.txt
LICENSE
```

## Data

The experiments use the public **CircuitNet-N14** dataset
(https://circuitnet.github.io , https://github.com/circuitnet/CircuitNet).

Each sample is preprocessed into:
- feature: a 3-channel array `(macro-region, RUDY, RUDY-pin)` saved as `.npy`
- label:   the combined (max of horizontal/vertical) global-routing overflow map

Place the stacked dataset at `cn_congestion/{feature,label}/*.npy`. Filenames encode
the design as a prefix before `__` (e.g. `RISCY-FPU__<id>.npy`), which the cross-design
and generalisation scripts use to split by design.

## How to reproduce

The scripts are written for **Google Colab** (GPU). Put `src/model.py` in the
runtime as `deeproute_c_colab.py` (the scripts reference
`/content/deeproute_c_colab.py`) and the dataset at `/content/cn_congestion`, then run,
for example:

```bash
python scripts/test_split_experiment.py      # within-design test-set factorial
python scripts/generalization_experiment.py  # cross-design generalisation
python scripts/detection_metrics.py          # detection / false-positive metrics
```

Each script writes a CSV of results (and, if Google Drive is mounted, copies it there).
Random seeds are fixed; GPU/library versions are printed at run time.

## Requirements

See `requirements.txt`. Core: PyTorch, NumPy, scikit-image, scikit-learn, SciPy,
huggingface_hub (for downloading CircuitNet tarballs).

## Citation

```
[CONFIRM — add the paper citation / DOI once published]
```

## License

MIT (see LICENSE).
