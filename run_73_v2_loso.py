"""Run the existing leakage-safe 73-station protocol with the opt-in V2 model.

Only final per-station summaries/predictions, six small selector locks, and the
single active resume state are kept on Drive. Selector checkpoints are temporary.
"""
from __future__ import annotations

import os


def main() -> None:
    os.environ["DL_TCN_MODEL_VERSION"] = "v2"
    os.environ.setdefault(
        "DL_TCN_GROUP_73_OUTPUT_ROOT",
        "/content/drive/MyDrive/DL_TCN_V2_73_GROUP_SELECTOR",
    )
    os.environ.setdefault("DL_TCN_MAX_EPOCHS", "15")
    os.environ.setdefault("DL_TCN_SAVE_EPOCH_PREDICTIONS", "1")
    from run_73_group_selector_loso import main as run
    run()


if __name__ == "__main__":
    main()
