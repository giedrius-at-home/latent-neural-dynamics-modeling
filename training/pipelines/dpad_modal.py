"""Modal orchestration for the DPAD pipeline.

Two-stage fan-out (config-driven; one container per stage entry):

  1. ``train_one(config_path, model_dbs)`` -> --phases train
  2. ``run_post_train(config_path)``       -> --phases predictions,forecasts,classification
     Runs after stage 1 has fully drained for each config.

Mode strings match experiment.type in YAML: z-as-neural | z-as-behavior.
Config files are named: dpad_{participant}_S{session}_{mode}.yaml

Sweeps discover configs by globbing a directory -- no hardcoded sessions or modes.
Organize configs into directories by sweep scope (e.g., training/setups/dpad_modal/).

Usage via unified pipeline (single-session YAML, uses experiment.type for mode filter):
    python -m training.pipeline --config training/setups/dpad_modal/my_config.yaml

Usage via modal run directly (multi-session sweeps):
    modal run training/pipelines/dpad_modal.py::check
    modal run training/pipelines/dpad_modal.py::sweep --configs-dir training/setups/dpad_modal
    modal run training/pipelines/dpad_modal.py::sweep --configs-dir training/setups/dpad_modal --mode z-as-neural
    modal run training/pipelines/dpad_modal.py::sweep --configs-dir training/setups/dpad_modal --mode z-as-behavior
"""

import os
import subprocess
import sys
from pathlib import Path

import modal
import yaml

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    PROJECT_ROOT = Path("/app")

_image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu")
    .apt_install("git", "build-essential")
    # Step 1: install most packages from the exact conda env export.
    # DPAD 0.0.9 pins PSID==1.2.5 as a dep; PSID is intentionally absent here
    # so pip installs 1.2.5 as DPAD's transitive dep without conflict.
    .pip_install_from_requirements(str(PROJECT_ROOT / "modal_requirements.txt"))
    # Step 2: overwrite with PSID==1.2.6 (our version) ignoring DPAD's pin,
    # and add torch + scipy which are excluded from the conda export.
    .run_commands(
        "pip install --no-deps PSID==1.2.6",
        "pip install torch==2.9.1 scipy==1.16.0",
    )
    .env({"TF_CPP_MIN_LOG_LEVEL": "2"})
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path="/app",
        ignore=[
            ".git/**",
            "**/__pycache__/**",
            ".venv/**",
            "**/.ipynb_checkpoints/**",
            "results/**",
            "resampled_recordings/**",
            "training/setups/**",
            "classification/setups/**",
            "thesis_figures/**",
            "archives/**",
            "notebooks/**",
            "logs/**",
            "data/**",
            "reports/pipeline_runs/**",
            ".pytest_cache/**",
            "node_modules/**",
            "*.log",
            "**/*.log",
            "*.png",
            "*.pdf",
            "*.html",
        ],
    )
)

_data_vol = modal.Volume.from_name("dpad-data", create_if_missing=True)
_results_vol = modal.Volume.from_name("dpad-results", create_if_missing=True)
_training_vol = modal.Volume.from_name("dpad-training-setups", create_if_missing=True)
_VOLUMES = {
    "/app/resampled_recordings": _data_vol,
    "/app/results": _results_vol,
    "/app/training/setups": _training_vol,
}

app = modal.App("dpad-pipeline")


def _run_dpad(config_path: str, phases: str, model_dbs: str = None) -> None:
    os.chdir("/app")
    cmd = [
        "python",
        "-m",
        "training.pipeline",
        "--config",
        config_path,
        "--phases",
        phases,
    ]
    if model_dbs:
        cmd += ["--dbs", model_dbs]
    print(">>> ", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    _results_vol.commit()
    _training_vol.commit()


def _load_sweep_entries(configs_dir: str, mode_filter: str = None) -> list:
    """Return [(config_path, sides)] for each YAML in configs_dir matching mode_filter."""
    entries = []
    for p in sorted(Path(configs_dir).glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        mode = cfg.get("experiment", {}).get("type", "z-as-neural")
        if mode_filter and mode != mode_filter:
            continue
        sides = (
            cfg.get("experiment", {})
            .get("train", {})
            .get("model_dbs_state", ["both", "on", "off"])
        )
        entries.append((str(p), list(sides)))
    return entries


@app.function(image=_image, gpu="A10G", timeout=86400, volumes=_VOLUMES)
def train_one(config_path: str, model_dbs: str) -> str:
    """Stage 1: train one config x one dbs side in isolation."""
    _run_dpad(config_path, phases="train", model_dbs=model_dbs)
    return f"trained:{config_path}:{model_dbs}"


@app.function(image=_image, gpu="A10G", timeout=86400, volumes=_VOLUMES)
def run_post_train(config_path: str, phases: str = "predictions,forecasts,classification") -> str:
    """Stage 2: post-train phases for one config (default: all non-train phases)."""
    _run_dpad(config_path, phases=phases)
    return f"post_train:{config_path}:{phases}"


@app.function(image=_image, gpu="A10G", timeout=600, volumes=_VOLUMES)
def gpu_check() -> None:
    import tensorflow as tf  # image-side dep

    subprocess.run(["nvidia-smi"], check=False)
    print("TF version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))


def _run_sweep(label: str, entries: list, phases: str = "") -> None:
    """Two-stage fan-out: drain all train containers before spawning post-train."""
    run_train = not phases or "train" in phases
    run_post = not phases or any(
        p in phases for p in ("predictions", "forecasts", "classification")
    )

    if run_train:
        train_args = [(cfg, side) for (cfg, sides) in entries for side in sides]
        print(f"[{label}] 1/2 train: {len(train_args)} containers ...")
        for r in train_one.starmap(train_args):
            print("  ", r)

    if run_post:
        # Forward exactly the requested post-train phases (drop 'train'); the
        # container fn no longer hardcodes predictions,forecasts,classification.
        post_phase_list = [
            p for p in (phases.split(",") if phases else
                        ["predictions", "forecasts", "classification"])
            if p.strip() and p.strip() != "train"
        ]
        post_phases = ",".join(post_phase_list)
        post_args = [(cfg, post_phases) for (cfg, _) in entries]
        print(f"[{label}] 2/2 post-train ({post_phases}): {len(post_args)} containers ...")
        for r in run_post_train.starmap(post_args):
            print("  ", r)


@app.local_entrypoint()
def check():
    gpu_check.remote()


@app.local_entrypoint()
def sweep(configs_dir: str, mode: str = "", phases: str = ""):
    """Sweep all configs in configs_dir.

    --mode z-as-neural | z-as-behavior  filter to one experiment type (default: all)
    --phases train,predictions,...       subset of pipeline phases (default: all)
    """
    entries = _load_sweep_entries(configs_dir, mode_filter=mode or None)
    _run_sweep("sweep", entries, phases)


@app.local_entrypoint()
def sweep_spawn(configs_dir: str, mode: str = "", phases: str = "forecasts,classification"):
    """Fire-and-forget post-train sweep via ``.spawn()``.

    Unlike ``sweep`` (blocking ``.starmap`` that ties container lifetime to the
    local process), this spawns each container server-side and returns at once,
    so the run survives the local client exiting or disconnecting. Poll the
    results volume / ``modal app list`` for completion. Post-train phases only
    (drops any 'train'); use ``sweep`` if you need training.
    """
    entries = _load_sweep_entries(configs_dir, mode_filter=mode or None)
    post_phase_list = [
        p for p in phases.split(",") if p.strip() and p.strip() != "train"
    ]
    post_phases = ",".join(post_phase_list)
    print(f"[sweep_spawn] spawning {len(entries)} containers ({post_phases}) ...")
    calls = [run_post_train.spawn(cfg, post_phases) for (cfg, _) in entries]
    for cfg, call in zip((c for c, _ in entries), calls):
        print(f"  spawned {call.object_id}  {Path(cfg).name}")
    print("[sweep_spawn] all spawned; containers run server-side. Safe to exit.")


class DPADModalPipeline:
    """Wraps the Modal fan-out sweep; invoked by ``training.pipeline`` when
    ``framework.name == "dpad_modal"``.

    Calls ``sweep`` on the config's parent directory, filtered to the config's
    own experiment.type. Put configs for a given run in a dedicated directory
    (e.g., training/setups/dpad_modal/) so the sweep picks up exactly the right set.
    """

    def __init__(self, config, log, phases=None, dbs=None):
        self.config = config
        self.log = log
        self.phases = phases

    def run(self) -> None:
        config_path = Path(
            object.__getattribute__(self.config, "_source_path").resolve()
        )
        exp_type = self.config.experiment.type
        cmd = [
            "modal",
            "run",
            "training/pipelines/dpad_modal.py::sweep",
            "--configs-dir",
            str(config_path.parent),
            "--mode",
            exp_type,
        ]
        if self.phases:
            cmd += ["--phases", ",".join(self.phases)]
        self.log.info("Modal sweep: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
