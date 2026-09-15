# AeroCast 73 Lab

臺灣 73 個空氣品質測站的 unseen-station cold-start PM2.5 nowcast。

目前保留兩個深度學習版本：

- V1：shared causal TCN + target-conditioned cross-attention。
- V2：加入 train-only background/anomaly、49 項 donor-target static 差異、
  physics-guided attention、station-pair virtual target 與高污染 tail-aware
  loss。V2 **不使用 target 氣象或 target dynamic history**。
- V7 pilot：context-held-out gradient operator。詳細規格與一格 Colab
  執行方式見 [V7_README.md](V7_README.md)。

完整 V2 方法說明見 [V2_README.md](V2_README.md)。

## Colab 安裝

先掛載 Google Drive，並把資料 ZIP 解壓成：

```text
/content/dl_tcn_data/
  AQX_P_15_Resource/
  resources/
    station_static_features_49.csv
    station_static_clusters.csv
```

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!git clone https://github.com/Kadar7/aerocast-73-lab.git dl_tcn_cross_attention
%cd /content/dl_tcn_cross_attention
!pip install -q -r requirements-colab.txt
!python verify_colab_setup.py
```

若資料 ZIP 尚未解壓：

```python
import zipfile
with zipfile.ZipFile("/content/dl_tcn_colab_data.zip") as archive:
    archive.extractall("/content")
```

## 執行 V2

以下是 A100 設定；L4 請改成 batch 512、workers 4、compile off。

```python
import os

os.environ["DL_TCN_DATA_ROOT"] = "/content/dl_tcn_data"
os.environ["DL_TCN_BATCH_SIZE"] = "1024"
os.environ["DL_TCN_NUM_WORKERS"] = "8"
os.environ["DL_TCN_COMPILE_MODE"] = "reduce-overhead"
os.environ["DL_TCN_MAX_NEW_STATIONS"] = "1"  # 先完成一個outer target

!python -u run_73_v2_loso.py
```

第一站確認正常後：

```python
os.environ["DL_TCN_MAX_NEW_STATIONS"] = "0"
!python -u run_73_v2_loso.py
```

重連後執行相同程式即可。已完成 targets 會跳過；目前進行中的
selector/refit 會從單一 resume state 繼續。

V2 預設輸出：

```text
/content/drive/MyDrive/DL_TCN_V2_73_GROUP_SELECTOR/
```

只保留每站最終 summary、壓縮 prediction、六個 selector locks，以及當前
resume state。暫存 selector snapshots 完成後會刪除。

## 快速檢查

不啟動正式訓練：

```python
!python tests/test_v2_synthetic.py
!python tests/smoke_v2_real_data.py
```

V1 的 73 站入口仍為：

```python
!python -u run_73_group_selector_loso.py
```

資料、checkpoint、CSV 結果與 Google Drive 輸出不存放在此公開 repository。
