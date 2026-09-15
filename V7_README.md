# V7 CHO-GO pilot

V7 使用 context-held-out gradient operator。每個 donor 的實測 PM2.5
只在最後作為 absolute anchor；估計該 donor 到 target 的 potential difference
時，donor 所屬 hash group 的全部動態資料已從 context 移除。

固定規格：

- unseen-station cold start；outer target 全時段排除。
- 60 training / 12 validation；training donor 59、validation donor 60。
- 24 小時 causal history、static49、9 raw channels 與衍生 wind along/cross。
- 不使用 target dynamic、target weather、background、MLDG、LGBM 或 epoch selector。
- fold 0、3 各跑 5 epochs；這是探索性 performance pilot。

## Colab 一格執行

先選 A100 或 L4 runtime，掛載 Drive，並把資料 ZIP 上傳至 `/content`。

```python
from google.colab import drive
drive.mount('/content/drive')

import os, pathlib, shutil, subprocess, sys, zipfile

repo = pathlib.Path('/content/aerocast-73-lab')
if repo.exists():
    shutil.rmtree(repo)
subprocess.run(['git','clone','https://github.com/Kadar7/aerocast-73-lab.git',str(repo)],check=True)

data = pathlib.Path('/content/dl_tcn_data')
if not data.exists():
    archives=list(pathlib.Path('/content').glob('dl_tcn_colab_data*.zip'))
    if len(archives)!=1:
        raise FileNotFoundError(f'需要一個資料ZIP，找到：{archives}')
    with zipfile.ZipFile(archives[0]) as z:
        z.extractall('/content')
if not (data/'AQX_P_15_Resource').is_dir():
    raise FileNotFoundError(data/'AQX_P_15_Resource')

subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(repo/'requirements-colab.txt')],check=True)
os.environ.update({
    'DL_TCN_DATA_ROOT': str(data),
    'DL_TCN_WORK_ROOT': '/content/dl_tcn_work',
    'DL_TCN_TARGET_SITE': '桃園',
    'DL_TCN_COMPILE_MODE': 'off',
    'V7_FOLDS': '0,3',
    'V7_EPOCHS': '5',
    'V7_BATCH_SIZE': '16',
    'V7_VALIDATION_BATCH_SIZE': '16',
    'V7_STEPS_PER_EPOCH': '400',
    'V7_OUTPUT_ROOT': '/content/DL_TCN_V7_CHOGO_PILOT',
    'V7_DRIVE_OUTPUT': '/content/drive/MyDrive/DL_TCN_V7_CHOGO_PILOT',
})
subprocess.run([sys.executable,str(repo/'verify_colab_setup.py')],cwd=repo,check=True)
subprocess.run([sys.executable,str(repo/'tests/test_v7_chogo.py')],cwd=repo,check=True)
subprocess.run([sys.executable,'-u',str(repo/'run_v7_chogo_pilot.py')],cwd=repo,check=True)
```

訓練過程不寫檔。兩個 folds 全部成功後才會一次生成 `pilot_summary.md`、
`pilot_summary.json`、best checkpoint、
逐站指標與壓縮 predictions，並同步到：

`/content/drive/MyDrive/DL_TCN_V7_CHOGO_PILOT`

如果顯存不足，只降低 `V7_BATCH_SIZE`；不要修改模型寬度。若要延長完整
pilot，再把 `V7_EPOCHS` 改成 15 或 25。
