# V8 SASFO budget-gated pilot

V8 is a new model, not a renamed V7. It replaces V7's quadratic donor graph and four held-out-group forwards with one linear-time signed affine donor-field operator.

## Locked pilot protocol

- Outer target: 桃園, completely hidden.
- Folds: 0 and 3 only; each is 60 training stations / 12 validation stations.
- Training target donors: 59. Validation target donors: 60.
- Inputs remain 49 static features, 11 dynamic channels, 24 causal hours, and the reviewed geometry.
- Dynamic missing values remain zero placeholders plus masks; there is no imputation.
- Batch: exactly 4 stations × 64 timestamps = 256.
- One epoch: 128 complete station rounds = 1,920 optimizer steps and 8,192 samples per training station.
- Run 15 complete epochs, record the state, then always run epochs 16–18. If epochs 16–18 still improve macro station RMSE by at least 0.25% over the best result through epoch 15, continue through epoch 20; otherwise stop at 18. Best and last/resume models are both retained, so a later explicitly authorized run can continue from epoch 21.
- Loss: station-balanced by construction, normalized only by the train-only global PM2.5 standard deviation.
- The benchmark runs first on both 59/60-donor and 71/72-donor paths. It reports conservative costs for the two-fold pilot, six-fold selection, 73 refits, 20% recovery reserve and a five-hour analysis margin. A projected pilot above four A100 hours, or a complete plan above the available 60 A100 hours, aborts before training so the implementation can be optimized without reducing training exposure.
- The active checkpoint is overwritten every 100 steps and copied to Drive. Every epoch receives full 12-station validation. Completed folds save best model, last/resume model, histories and compact best predictions immediately and are skipped on resume.
- The script never launches the 73-station experiment.

## Gate

V8 is `GO` only if it passes all four predeclared comparisons against a separately supplied, protocol-matched V6 fold-0/fold-3 artifact. If that exact artifact is absent, the pilot still reports V8 results but is labelled `BASELINE_UNAVAILABLE`, not `GO`:

1. each fold macro RMSE improves by at least 5%;
2. each fold mean absolute station bias does not worsen;
3. each fold sparse-quartile macro RMSE does not worsen;
4. pooled exact R² across both folds improves by at least 0.04.

Any failure is `NO_GO`; there is no automatic retry, tuning, or full run.

## Colab: one cell

```python
from google.colab import drive
drive.mount('/content/drive')

import os, pathlib, subprocess, sys, zipfile

repo = pathlib.Path('/content/aerocast-73-lab')
if not repo.exists():
    subprocess.run(['git', 'clone', 'https://github.com/Kadar7/aerocast-73-lab.git', str(repo)], check=True)
else:
    subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)

data_zip = pathlib.Path('/content/dl_tcn_colab_data.zip')
data_root = pathlib.Path('/content/dl_tcn_data')
if not data_root.exists():
    if not zipfile.is_zipfile(data_zip):
        raise RuntimeError('Please upload a valid /content/dl_tcn_colab_data.zip')
    with zipfile.ZipFile(data_zip) as archive:
        archive.extractall('/content')

os.environ['DL_TCN_DATA_ROOT'] = str(data_root)
os.environ['DL_TCN_TARGET_SITE'] = '桃園'
os.environ['V8_OUTPUT_ROOT'] = '/content/DL_TCN_V8_SASFO_PILOT'
os.environ['V8_DRIVE_OUTPUT'] = '/content/drive/MyDrive/DL_TCN_V8_SASFO_PILOT'
os.environ['DL_TCN_COMPILE_MODE'] = 'off'

subprocess.run([sys.executable, '-u', str(repo / 'run_v8_sasfo_pilot.py')], cwd=repo, check=True)
```

Important outputs in Drive are limited to `runtime_profile.json`, one active resume checkpoint while running, two completed fold folders, and `pilot_gate.json`.
