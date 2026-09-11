from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
IS_COLAB = Path("/content").is_dir()
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "DL_TCN_DATA_ROOT", "/content/dl_tcn_data" if IS_COLAB else str(SCRIPT_DIR / "dl_tcn_data")
))
DEFAULT_WORK_ROOT = Path(os.environ.get(
    "DL_TCN_WORK_ROOT", "/content/dl_tcn_work" if IS_COLAB else str(SCRIPT_DIR / "dl_tcn_work")
))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get(
    "DL_TCN_OUTPUT_ROOT", "/content" if IS_COLAB else str(SCRIPT_DIR)
))

@dataclass
class Config:
    data_root: Path = DEFAULT_DATA_ROOT
    aq_data_dir: Path = DEFAULT_DATA_ROOT / "AQX_P_15_Resource"
    target_site: str = os.environ.get("DL_TCN_TARGET_SITE", "桃園")
    script_dir: Path = SCRIPT_DIR
    resource_dir: Path = DEFAULT_DATA_ROOT / "resources"
    static_path: Path = resource_dir / "station_static_features_49.csv"
    cluster_path: Path = resource_dir / "station_static_clusters.csv"
    output_dir: Path = DEFAULT_WORK_ROOT
    formal_output_root: Path = DEFAULT_OUTPUT_ROOT

    train_start: str = "2024-07-01 00:00:00"
    train_end: str = "2025-06-30 23:00:00"
    test_start: str = "2025-07-01 00:00:00"
    test_end: str = "2026-06-30 23:00:00"
    validation_stations: int = 12
    seed: int = 42

    history_hours: int = 24
    raw_dynamic_items: tuple[str, ...] = (
        "SO2", "CO", "O3", "PM10", "NO", "NO2", "AMB_TEMP", "PM2.5", "RH",
    )
    derived_dynamic_items: tuple[str, ...] = ("WIND_ALONG", "WIND_CROSS")
    aq_cube_items: tuple[str, ...] = (
        "SO2", "CO", "O3", "PM10", "NO", "NO2",
        "WIND_SPEED", "WIND_DIREC", "AMB_TEMP", "PM2.5", "RH",
    )

    tcn_hidden: int = 32
    tcn_kernel_size: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = float(os.environ.get("DL_TCN_DROPOUT", "0.20"))
    attention_dim: int = 32
    attention_heads: int = 2
    final_hidden: int = 32

    # Runtime profile may raise these values on Colab/L4. Environment variables
    # always win, which makes OOM recovery possible without editing the code.
    batch_size: int = int(os.environ.get("DL_TCN_BATCH_SIZE", "64"))
    num_workers: int = 0
    learning_rate: float = float(os.environ.get("DL_TCN_LEARNING_RATE", "5e-4"))
    weight_decay: float = float(os.environ.get("DL_TCN_WEIGHT_DECAY", "1e-4"))
    loss_name: str = "station_balanced_mse"
    gradient_clip_norm: float = float(os.environ.get("DL_TCN_GRADIENT_CLIP_NORM", "5.0"))
    smoke_epochs: int = 2
    smoke_max_train_batches: int = 20
    smoke_max_val_batches: int = 10
    use_amp: bool = True
    prefer_bf16: bool = True
    enable_tf32: bool = True
    amp_init_scale: float = 128.0
    amp_growth_interval: int = 1_000_000
    cache_schema_version: str = "masked_tcn_ca_v2_exact_pair_wind_scaler"

    # Formal 60/12 supervised training (no outer test/refit/search).
    max_epochs: int = int(os.environ.get("DL_TCN_MAX_EPOCHS", "15"))
    early_stopping_patience: int = int(os.environ.get("DL_TCN_PATIENCE", "5"))
    formal_output_dirname: str = "DL_TCN_CA_v1_formal_lr5e-4_station_balanced_output"
    outer_output_dirname: str = "DL_TCN_CA_v1_outer_60donors_output"
    progress_every_batches: int = 1000
    formal_num_workers: int = int(os.environ.get("DL_TCN_NUM_WORKERS", "0"))
    prefetch_factor: int = int(os.environ.get("DL_TCN_PREFETCH_FACTOR", "4"))
    l4_batch_size: int = 512
    l4_num_workers: int = 4
    smoke_batch_size: int = 8
    gpu_precompute_chunk_hours: int = 512
    # "auto" enables the benchmarked reduce-overhead graph on A100 only.
    # Set DL_TCN_COMPILE_MODE=off to disable it for troubleshooting.
    compile_mode: str = os.environ.get("DL_TCN_COMPILE_MODE", "auto").strip().lower()

    @property
    def dynamic_items(self) -> tuple[str, ...]:
        return self.raw_dynamic_items + self.derived_dynamic_items

    @property
    def n_dynamic_channels(self) -> int:
        return len(self.dynamic_items)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config()


def apply_runtime_profile(cfg: Config = CFG) -> dict:
    """Tune execution only; do not change data, protocol, model, or loss."""
    cpu_count = os.cpu_count() or 1
    profile = {
        "name": "cpu",
        "cpu_count": cpu_count,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.formal_num_workers,
        "amp_dtype": "disabled",
        "gpu_name": None,
        "gpu_vram_gb": 0.0,
        "compile_mode": "off" if cfg.compile_mode == "auto" else cfg.compile_mode,
    }
    if not torch.cuda.is_available():
        if cfg.compile_mode == "auto":
            cfg.compile_mode = "off"
        return profile

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    is_l4 = "L4" in gpu_name.upper()
    is_a100 = "A100" in gpu_name.upper()
    if "DL_TCN_BATCH_SIZE" not in os.environ:
        cfg.batch_size = 1024 if is_a100 else (cfg.l4_batch_size if is_l4 else (256 if vram_gb >= 14 else (128 if vram_gb >= 8 else 64)))
    if "DL_TCN_NUM_WORKERS" not in os.environ:
        cfg.formal_num_workers = min(cfg.l4_num_workers if is_l4 else 2, max(cpu_count - 1, 0))

    compute_capability = torch.cuda.get_device_capability(0)
    bf16 = bool(
        cfg.use_amp and cfg.prefer_bf16 and compute_capability[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    if cfg.compile_mode == "auto":
        cfg.compile_mode = "reduce-overhead" if is_a100 else "off"
    if cfg.compile_mode not in {"off", "default", "reduce-overhead"}:
        raise ValueError("DL_TCN_COMPILE_MODE只能是auto/off/default/reduce-overhead")
    profile.update({
        "name": "colab_a100" if is_a100 else ("colab_l4" if is_l4 else "cuda_auto"),
        "gpu_name": gpu_name,
        "gpu_vram_gb": vram_gb,
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
        "batch_size": cfg.batch_size,
        "num_workers": cfg.formal_num_workers,
        "amp_dtype": "bfloat16" if bf16 else ("float16" if cfg.use_amp else "disabled"),
        "compile_mode": cfg.compile_mode,
    })
    return profile
