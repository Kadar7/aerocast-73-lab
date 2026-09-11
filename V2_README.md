# TCN Cross-Attention V2

V2 keeps the original unseen-station LOSO protocol and adds five opt-in
components. It does **not** use target-side meteorology or target PM2.5 during
selection/training.

1. A train-only long-term background (static ridge + geographic IDW). Source
   target backgrounds are leave-one-station-out; validation/outer backgrounds
   use source stations only. The network predicts an anomaly on this baseline.
2. Explicit donor minus target differences for all 49 standardized static
   features.
3. A physics-guided attention prior using distance, donor wind projected along
   donor-to-target bearing, cross-wind magnitude, and static similarity.
4. A low-weight increment-inspired station-pair virtual-target regularizer.
   It interpolates source station pairs because the supplied data contain
   station static features, not a full arbitrary-location GIS feature grid.
5. Station-balanced Huber loss with a smooth PM2.5 tail weight and a >=35
   microgram/m3 event auxiliary head.

Missing dynamic values remain exactly as V1: no imputation, standardized zero
placeholder plus binary mask. The target query contains only target static,
calendar time, and the source-fitted background.

## Colab: run all 73 outer targets

```python
%cd /content/dl_tcn_cross_attention
!git pull

import os
os.environ["DL_TCN_DATA_ROOT"] = "/content/dl_tcn_data"
os.environ["DL_TCN_BATCH_SIZE"] = "1024"       # A100; use 512 on L4
os.environ["DL_TCN_NUM_WORKERS"] = "8"         # A100/12 CPU; use 4 on L4
os.environ["DL_TCN_COMPILE_MODE"] = "reduce-overhead"  # A100; use off on L4
os.environ["DL_TCN_MAX_NEW_STATIONS"] = "0"    # 0 = all remaining stations

!python -u run_73_v2_loso.py
```

Rerun the same cell after a disconnect. Completed targets are skipped and the
single active selector/refit state resumes. Drive output defaults to
`/content/drive/MyDrive/DL_TCN_V2_73_GROUP_SELECTOR`.

Only the existing compact final station summary/prediction files, six selector
locks, and one active resume state are retained. Temporary selector snapshots
are deleted after a lock is produced.

## Optional V2 weights

Defaults are deliberately conservative and can be changed before execution:

```python
os.environ["DL_TCN_V2_HUBER_BETA"] = "5.0"
os.environ["DL_TCN_V2_TAIL_START"] = "25.0"
os.environ["DL_TCN_V2_EVENT_THRESHOLD"] = "35.0"
os.environ["DL_TCN_V2_TAIL_MAX_WEIGHT"] = "3.0"
os.environ["DL_TCN_V2_EVENT_LOSS_WEIGHT"] = "0.1"
os.environ["DL_TCN_V2_VIRTUAL_LOSS_WEIGHT"] = "0.1"
```

For a cheap first check, set `DL_TCN_MAX_NEW_STATIONS=1`. Use a new output
directory when changing any scientific setting; do not resume an old V2 run
with different loss settings.
