import os
from pathlib import Path
import torch
from config import CFG, apply_runtime_profile

profile=apply_runtime_profile(CFG)

required=(CFG.static_path,CFG.cluster_path)
missing=[str(path) for path in required if not path.is_file()]
aq_files=sorted(CFG.aq_data_dir.glob("*.csv")) if CFG.aq_data_dir.is_dir() else []
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"gpu_vram_gb={profile['gpu_vram_gb']:.1f}")
print(f"cpu_count={os.cpu_count()}")
print(f"runtime_profile={profile['name']}")
print(f"planned_batch_size={profile['batch_size']}")
print(f"planned_num_workers={profile['num_workers']}")
print(f"planned_amp_dtype={profile['amp_dtype']}")
print(f"data_root={CFG.data_root}")
print(f"aq_csv_count={len(aq_files)}")
print(f"static_path={CFG.static_path}")
print(f"cluster_path={CFG.cluster_path}")
print(f"formal_output_root={CFG.formal_output_root}")
if missing: raise FileNotFoundError(f"缺少資料檔: {missing}")
if len(aq_files)!=72: raise RuntimeError(f"AQ CSV應為72份，實際{len(aq_files)}")
if not torch.cuda.is_available(): raise RuntimeError("Colab沒有啟用GPU runtime")
print("COLAB SETUP PASSED")
