# Unseen-station PM2.5 TCN + cross-attention

V2（background/anomaly、static差異、物理attention、虛擬target與高污染loss）
請看 [V2_README.md](V2_README.md)；入口是 `run_73_v2_loso.py`。原V1入口與結果不受影響。

這是已通過protocol review與pipeline等價測試的Colab版本。固定設定為60 training stations／12 validation stations、24小時history、11個dynamic channels、49個static features、shared causal TCN與target-conditioned cross-attention。`train_formal.py`只做60/12模型選擇；`evaluate_outer.py`在模型選擇鎖定後，讀取最佳checkpoint並對1個outer target做一次正式test。此階段不做72站refit，也不搜尋超參數。

## Colab快速執行

先在Colab選擇GPU runtime，然後依序執行：

```python
%cd /content
!unzip -q /content/dl_tcn_colab_data.zip -d /content
!git clone YOUR_GITHUB_REPOSITORY_URL dl_tcn_cross_attention
%cd /content/dl_tcn_cross_attention
!pip install -q -r requirements-colab.txt
!python verify_colab_setup.py
```

資料ZIP解壓後必須形成：

```text
/content/dl_tcn_data/
  AQX_P_15_Resource/       # 72 CSV
  resources/
    station_static_features_49.csv
    station_static_clusters.csv
```

正式訓練前可選擇跑少量新舊pipeline等價測試：

```python
!python tests/verify_pipeline_equivalence.py
```

開始正式training：

```python
!python train_formal.py
```

目前鎖定config：learning rate `5e-4`、weight decay `1e-4`、dropout `0.20`、gradient clip `5.0`、station-balanced MSE、max epochs `15`、patience `5`。station-balanced權重依60個training stations各自可用樣本數計算，使每站對期望loss的貢獻相同。Colab L4會自動使用batch size `512`、4個輕量index workers、BF16 AMP、TF32與fused AdamW；其他CUDA GPU會依VRAM保守選擇batch。資料、split、特徵與模型架構不變。

CUDA正式訓練會在每次run開始時，一次性預計算9個raw channels的mask/標準化值，以及所有target-donor的WIND_ALONG/WIND_CROSS；每個epoch只擷取24小時窗口，不重算三角函數，也不預展開全年sample tensor。啟動時會印出預計算秒數與table VRAM。

如L4仍發生OOM，不需改程式，可在正式訓練前覆寫：

```python
%env DL_TCN_BATCH_SIZE=256
```

其他常用訓練設定也可在啟動程式前覆寫：

```python
%env DL_TCN_LEARNING_RATE=5e-4
%env DL_TCN_WEIGHT_DECAY=1e-4
%env DL_TCN_MAX_EPOCHS=15
%env DL_TCN_PATIENCE=5
```

## 建議將輸出寫入Google Drive

Colab的`/content`會在runtime結束後消失。若要保存checkpoint與history，先掛載Drive並指定輸出根目錄：

```python
from google.colab import drive
drive.mount('/content/drive')
%env DL_TCN_OUTPUT_ROOT=/content/drive/MyDrive/dl_tcn_runs
!python train_formal.py
```

AQ memory-map cache預設留在`/content/dl_tcn_work`，避免將頻繁cache I/O寫到Drive。可用環境變數覆寫：

- `DL_TCN_DATA_ROOT`
- `DL_TCN_WORK_ROOT`
- `DL_TCN_OUTPUT_ROOT`

## 正式輸出

輸出資料夾為`DL_TCN_CA_v1_formal_lr5e-4_station_balanced_output`，包含：

- `best_checkpoint.pt`
- `training_history.csv`
- `training_summary.json`
- `validation_predictions.csv`
- `best_validation_station_metrics.csv`
- `validation_station_metrics_all_epochs.csv`
- `pretraining_sanity.json`

## Outer test（60/12/1，不refit）

模型選擇完成後，將`DL_TCN_SELECTION_CHECKPOINT`指向鎖定實驗的`best_checkpoint.pt`，再執行：

```bash
python evaluate_outer.py
```

程式會核對checkpoint的Reduced cluster-aware 60/12 split與train-only scaler。Outer target完全不參與training、validation、scaler或epoch選擇；outer推論只使用60個training stations，12個validation stations不會混入donor pool。輸出資料夾`DL_TCN_CA_v1_outer_60donors_output`包含逐時預測、整體指標、coverage分母與protocol audit。72站refit是另一個後續階段，本程式不會執行。

## Target-aware checkpoint selection離線分析

`analyze_target_aware_epoch.py`只讀取既有`training_history.csv`、`validation_station_metrics_all_epochs.csv`、`best_checkpoint.pt`與49項static features，不修改模型也不重新訓練。它使用checkpoint保存的60站train-only static scaler，對12個validation pseudo-target做leave-one-station-out static similarity加權epoch預測。每個target自己的best epoch不會進入其預測，PM2.5與future observation也不會進入similarity。

程式輸出station×epoch矩陣、similarity矩陣、PCA／heatmap、global／macro／target-aware／oracle逐站表現，以及明確列出12站樣本量與目前只保存best checkpoint的限制。

修正版同時保留原始49維RBF-all方法作失敗baseline，並加入60站train-only PCA、target對固定60 donors的距離／密度摘要、固定k=1/3/5近鄰及cluster-restricted候選。`target_aware_knn_nested_loso`會在每個held-out validation target之外，再用其餘11站做內層LOSO選擇相似度方法，因此連方法選擇也不會看到held-out target的best epoch。

曲線版`target_aware_curve_nested_loso`不再平均epoch編號，而是對每站完整的epoch-wise RMSE regret curve做regularized multi-output ridge預測，再對預測曲線取argmin。表示法與ridge強度同樣由不含held-out target的內層LOSO選擇。混合epoch方法的精確pooled R²使用既有`validation_predictions.csv`中的共同y_true SST與station×epoch RMSE重建，不再拿macro R²冒充overall R²。

## Target-conditioned snapshot ensemble

舊的static-similarity hard epoch方法只保留作失敗baseline，不再作為正式方案。新流程固定排除outer target，對其餘72站建立六個互斥的60-train/12-unseen-validation folds。Fold 0完全等於原本Reduced cluster-aware 60/12 split；六個folds合計讓每個known station恰好當一次unseen validation target。

```python
import os

os.environ["DL_TCN_DATA_ROOT"] = "/content/dl_tcn_data"
os.environ["DL_TCN_OUTPUT_ROOT"] = "/content"
os.environ["DL_TCN_CROSSFIT_ROOT"] = "/content/crossfit_target_conditioned_snapshots"
os.environ["DL_TCN_MAX_EPOCHS"] = "15"
os.environ["DL_TCN_BATCH_SIZE"] = "512"
os.environ["DL_TCN_LEARNING_RATE"] = "5e-4"
os.environ["DL_TCN_WEIGHT_DECAY"] = "1e-4"
os.environ["DL_TCN_DROPOUT"] = "0.20"

%cd /content/dl_tcn_cross_attention
!python train_crossfit_snapshots.py
!python fit_target_conditioned_snapshot_ensemble.py
```

訓練可續跑。每個fold完成後會建立`COMPLETE.json`；每個epoch也會保存`last_training_state.pt`。Colab中斷後重新執行同一格，已完成folds會跳過，未完成fold會從最後一個epoch繼續。也可用`DL_TCN_CROSSFIT_FOLDS="0,1"`只跑指定folds。

selector不再預測離散的`argmin(epoch)`，而是用72站OOF error curves直接學習各epoch的連續權重。完成後可執行正式outer ensemble：

```python
!python evaluate_outer_target_conditioned_ensemble.py
```

程式會先只用outer static鎖定權重並寫出`LOCKED_WEIGHTS_BEFORE_OUTER_TRUTH.json`，之後才建立outer test dataset和計分，避免用outer PM2.5選checkpoint。

## 72站粗分early／middle／late

`analyze_coarse_epoch_regime.py`讀取既有六折OOF epoch曲線，不重新訓練TCN。它將每站oracle epoch粗分為early（1–5）、middle（6–10）、late（11–15），使用原始49項static features分別訓練淺層CART與Extra Trees。

OOF評估時每次完整排除被預測站的類別與epoch曲線：分類器只用其餘71站，選定區間內的實際epoch也只依其餘71站平均RMSE決定。Bootstrap只評估分類穩定性。這是72站development analysis，不等同完全獨立outer驗證。

```python
%env DL_TCN_CROSSFIT_ROOT=/content/crossfit_target_conditioned_snapshots
%env DL_TCN_COARSE_BOOTSTRAPS=100
!python analyze_coarse_epoch_regime.py
```

主要結果位於`coarse_epoch_regime_analysis/coarse_regime_performance_summary.csv`；另輸出逐站分類、混淆矩陣、選定epoch指標、特徵重要性與CART規則。

## 透明static規則（不用分類器）

`analyze_interpretable_epoch_rules.py`不使用CART、Extra Trees或其他分類模型，只保留五項最有解釋力且較不重複的static features。固定門檻為：0–1 km森林比例≥1.43%、1–5 km森林比例≥12.53%、0–1 km交通比例≤17.89%、0–1 km商業比例≤2.39%、1–5 km建成比例≤17.60%。符合一項得1分；4–5分為early profile、2–3分為middle profile、0–1分為late profile。

六個fold是六次獨立訓練，因此raw epoch不直接跨fold平均。逐站檢查時，先以同fold另外11個validation站取得該次訓練的reference epoch，再將每站最佳epoch轉成fold-relative offset。被檢查站完全不參與自己的門檻判定以外之校準、fold reference或offset計算；同fold其他參考站的reference也會額外排除該target，因此由10站計算。目標依五項門檻分組後，程式比較同組station regret最低offset與同組最佳offset中位數，最後把offset加回該次訓練的reference epoch。輸出逐站列出五項數值、是否通過、總分、profile、reference epoch、真實／預測offset及RMSE regret。

```python
%env DL_TCN_CROSSFIT_ROOT=/content/crossfit_target_conditioned_snapshots
%env DL_TCN_MAX_EPOCHS=15
!python analyze_interpretable_epoch_rules.py
```

## 最佳化固定checkpoint規則樹

`evaluate_optimized_epoch_rule.py`使用已由72站搜尋完成的固定透明規則樹，共13個葉節點。規則使用交通比例、道路／工業區距離、森林比例、地形起伏、NDVI、經緯度與工業比例，輸出fold-relative offset。程式重現72站epoch MAE、分類正確率與RMSE後，會在完全不讀outer truth的情況下建立`LOCKED_OUTER_EPOCH_BEFORE_TRUTH.json`。

```python
%env DL_TCN_CROSSFIT_ROOT=/content/crossfit_target_conditioned_snapshots
%env DL_TCN_DATA_ROOT=/content/dl_tcn_data
!python evaluate_optimized_epoch_rule.py
```

## 組內細分checkpoint規則樹

`evaluate_refined_epoch_rule.py`沿用early／middle／late概念，再用static features細分為30個透明葉節點；每葉至少有2個known stations。輸出仍是相對於該次60/12訓練reference epoch的offset。程式先重現known-72開發結果，再只讀outer static並建立`LOCKED_OUTER_REFINED_EPOCH_BEFORE_TRUTH.json`，不讀outer PM2.5 truth。

```python
%env DL_TCN_CROSSFIT_ROOT=/content/crossfit_target_conditioned_snapshots
%env DL_TCN_DATA_ROOT=/content/dl_tcn_data
!python evaluate_refined_epoch_rule.py
```

規則鎖定後，才可對outer target解盲並計算完整epoch 1–15曲線：

```python
!python evaluate_outer_epoch_curve.py
```

輸出`outer_epoch_curve_metrics.csv`、壓縮逐時預測`outer_epoch_curve_predictions.npz`、`outer_locked_vs_oracle.json`與RMSE曲線`outer_epoch_curve_rmse.png`。`locked_epoch_before_truth`固定沿用上一階段的truth-free決策，不會被oracle曲線覆寫。

若只剩fold 0（例如Colab runtime被回收後復原），可設定`DL_TCN_REFINED_DEPLOY_ONLY=1`。此模式只讀fold 0的12站validation曲線計算reference epoch，不要求另外五折存在。

只為桃園復原fold 0時可設`DL_TCN_SAVE_EPOCH_PREDICTIONS=0`，略過不再需要的validation逐時NPZ；epoch checkpoints、validation metrics與resume state仍會完整保存。

精簡輸出模式不再建立每個epoch各一份station metrics；程式只維護單一`validation_station_metrics_all_epochs.csv`。成功完成fold後會移除已無用途的`last_training_state.pt`。Outer曲線預設只保存locked與oracle兩組逐時預測；只有明確設定`DL_TCN_SAVE_FULL_OUTER_CURVE_PREDICTIONS=1`才保存15個epoch的完整預測矩陣。

`benchmark_training_bottleneck.py`會用少量batch分別計時index loader、GPU feature assembly、forward/loss及backward/optimizer，輸出`training_bottleneck_profile.json`；它不保存或取代正式模型。
