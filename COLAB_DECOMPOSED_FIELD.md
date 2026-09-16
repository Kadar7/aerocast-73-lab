# Colab single-cell launcher

Select an A100 runtime, upload `dl_tcn_colab_data.zip` to `/content`, then run
this one cell. The pilot shows batch/epoch progress, resumes from Drive after a
disconnect, and automatically writes its compact final summary to Drive.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, pathlib, shutil, subprocess, sys, zipfile

repo = pathlib.Path('/content/aerocast-73-lab')
if repo.exists():
    subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)
else:
    subprocess.run(
        ['git', 'clone', 'https://github.com/Kadar7/aerocast-73-lab.git', str(repo)],
        check=True,
    )

data_root = pathlib.Path('/content/dl_tcn_data')
required = data_root / 'resources' / 'station_static_features_49.csv'
if not required.exists():
    candidates = [
        pathlib.Path('/content/dl_tcn_colab_data.zip'),
        pathlib.Path('/content/drive/MyDrive/dl_tcn_colab_data.zip'),
    ]
    archive = next((path for path in candidates if path.exists() and zipfile.is_zipfile(path)), None)
    if archive is None:
        raise FileNotFoundError(
            '找不到有效的dl_tcn_colab_data.zip；請上傳到/content，'
            '或放到Google Drive根目錄。'
        )
    with zipfile.ZipFile(archive) as package:
        package.extractall('/content')

    # Accept either a zip containing dl_tcn_data/ or the data contents directly.
    if not required.exists():
        found = list(pathlib.Path('/content').glob('**/station_static_features_49.csv'))
        found = [path for path in found if 'aerocast-73-lab' not in str(path)]
        if len(found) != 1:
            raise RuntimeError(f'解壓後無法唯一定位資料根目錄：{found}')
        source_root = found[0].parent.parent
        if data_root.exists():
            shutil.rmtree(data_root)
        shutil.move(str(source_root), str(data_root))

os.environ['DL_TCN_DATA_ROOT'] = str(data_root)
os.environ['DL_TCN_WORK_ROOT'] = '/content/dl_tcn_work'
os.environ['DL_TCN_COMPILE_MODE'] = 'off'
os.environ['DECOMPOSED_FIELD_OUTPUT'] = '/content/DL_DECOMPOSED_FIELD_PILOT'
os.environ['DECOMPOSED_FIELD_DRIVE_OUTPUT'] = '/content/drive/MyDrive/DL_DECOMPOSED_FIELD_PILOT'

subprocess.run(
    [sys.executable, str(repo / 'run_decomposed_field_pilot.py')],
    cwd=str(repo),
    env=os.environ.copy(),
    check=True,
)
```

The only Drive directory created is
`/content/drive/MyDrive/DL_DECOMPOSED_FIELD_PILOT`.  Before training it contains
`preflight.json`; during an unfinished fold it also contains one resumable
checkpoint. Completed folds retain only the best model, compact histories,
station metrics, compressed predictions and result JSON. The final decision is
in `pilot_summary.json`.
