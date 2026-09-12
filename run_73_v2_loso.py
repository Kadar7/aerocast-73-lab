"""Run the existing leakage-safe 73-station protocol with the opt-in V2 model.

Only final per-station summaries/predictions, six small selector locks, and the
single active resume state are kept on Drive. Selector checkpoints are temporary.
"""
from __future__ import annotations

import os
import json
from pathlib import Path


def main() -> None:
    os.environ["DL_TCN_MODEL_VERSION"] = "v2"
    os.environ.setdefault(
        "DL_TCN_GROUP_73_OUTPUT_ROOT",
        "/content/drive/MyDrive/DL_TCN_V4_FULL_DONOR_73",
    )
    os.environ.setdefault("DL_TCN_MAX_EPOCHS", "15")
    os.environ.setdefault("DL_TCN_SAVE_EPOCH_PREDICTIONS", "1")
    from v2_training import v2_config_payload
    root = Path(os.environ["DL_TCN_GROUP_73_OUTPUT_ROOT"])
    if str(root).startswith('/content/drive/') and not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Google Drive尚未掛載')
    marker = root / 'training_revision.json'
    settings = v2_config_payload()
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != settings:
            raise RuntimeError('Output root contains different training settings; use a new root')
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError('Existing unversioned results cannot be resumed as V4; use a new output root')
    else:
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    from run_73_group_selector_loso import main as run
    run()


if __name__ == "__main__":
    main()
