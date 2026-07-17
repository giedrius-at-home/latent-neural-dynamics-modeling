#!/usr/bin/env python3
"""Run DPAD h5 forecasts locally using the RTX 3050 GPU.

Patches each modal YAML to h_grid=[5.0] only, then calls
FrameworkPipeline(phases=('forecasts',)) inline (same process = GPU access).
Run background: nohup python3 notebooks/kaggle/run_dpad_forecasts_h5_local.py > /tmp/dpad_h5.log 2>&1 &
"""

import os
import sys
import tempfile
import re
from pathlib import Path

# Enable GPU memory growth before TF initializes so multiple parallel
# processes can share the 4GB VRAM without one process grabbing it all.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

SESSIONS = ["PDI1_S2", "PDI1_S4", "PDI4_S2", "PDI4_S3"]
MODES = ["z-as-behavior", "z-as-neural"]
H_VALUES = [5.0]

YAML_DIR = REPO / "training" / "setups" / "dpad_modal"

_H_RE = re.compile(r"(    h_grid:\n)(?:    - \S+\n)+")


def patch_yaml(text, h_values):
    block = "    h_grid:\n" + "".join(f"    - {h}\n" for h in h_values)
    return _H_RE.sub(block, text)


def run_combo(session, mode):
    from utils.config import get_config
    from utils.logger import setup_logging
    from training.pipelines._base import FrameworkPipeline

    yaml_path = YAML_DIR / f"dpad_modal_{session}_{mode}.yaml"
    if not yaml_path.exists():
        print(f"[SKIP] missing YAML: {yaml_path}")
        return

    patched = patch_yaml(yaml_path.read_text(), H_VALUES)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"dpad_{session}_{mode}_h5_",
        delete=False,
        dir="/tmp",
    ) as tmp:
        tmp.write(patched)
        tmp_path = Path(tmp.name)

    try:
        config = get_config(str(tmp_path))
        log = setup_logging(
            f"dpad_{session}_{mode}_h5",
            REPO / "logs" / "dpad" / f"{session}_{mode}_h5.log",
        )
        print(f"\n=== {session} {mode} h=5 ===", flush=True)
        FrameworkPipeline(config, log, phases=("forecasts",)).run()
        print(f"DONE {session} {mode}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    # Accept optional session/mode args for parallel invocation:
    #   python run_dpad_forecasts_h5_local.py PDI1_S2 z-as-behavior
    if len(sys.argv) == 3:
        session, mode = sys.argv[1], sys.argv[2]
        run_combo(session, mode)
    else:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        print(f"TF {tf.__version__} | GPUs: {gpus}")

        errors = []
        for session in SESSIONS:
            for mode in MODES:
                try:
                    run_combo(session, mode)
                except Exception as e:
                    errors.append((session, mode, str(e)))

        print(f"\n=== Finished. {len(errors)} errors: {errors} ===")
