"""Unified pipeline runner.

The pipeline type is declared in the config YAML via the ``framework:`` key:

    python -m training.pipeline --config training/setups/examples/psid_run.yaml
    python -m training.pipeline --config training/setups/examples/dpad_run.yaml
    python -m training.pipeline --config training/setups/examples/varma_run.yaml

Each YAML must contain a ``framework:`` section:

    framework:
      name: psid   # or dpad | dpad_modal | varma

All experiment parameters (DBS conditions, forecast grid, classification sweeps)
live in the ``experiment:`` section of the YAML.

Splits must be precomputed before running:

    python training/precompute_splits.py --participant PDI1 --session 2

For Modal cloud sweeps (DPAD only), use ``dpad_modal`` as the framework name:

    python -m training.pipeline --config training/setups/examples/dpad_run_on_modal.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training.pipelines._base import FrameworkPipeline
from training.pipelines.dpad_modal import DPADModalPipeline
from training.pipelines.psid_diagnostic import PsidDiagnosticPipeline
from utils.logger import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from utils.config import get_config

sys.path.insert(0, str(PROJECT_ROOT))


PIPELINES = {
    "psid": FrameworkPipeline,
    "dpad": FrameworkPipeline,
    "varma": FrameworkPipeline,
    "dpad_modal": DPADModalPipeline,
    "psid_diagnostic": PsidDiagnosticPipeline,
}


def run(
    config_path: str | Path, phases: str | None = None, dbs: str | None = None
) -> None:
    """Run the pipeline declared in ``config_path``.

    Args:
        config_path: Path to the run YAML (must contain ``framework.name``).
        phases: Comma-separated phase override, e.g. ``"train"`` or ``"predictions,forecasts"``.
        dbs: Single DBS condition to train (both | on | off). Overrides model_dbs_state in YAML.
    """
    config_path = Path(config_path)
    config = get_config(str(config_path))

    framework = config.framework.name
    Pipeline = PIPELINES.get(framework)
    if Pipeline is None:
        raise ValueError(
            f"Unknown framework {framework!r}; expected one of {list(PIPELINES)}"
        )

    log_path = (
        PROJECT_ROOT
        / config.results.log_dir
        / f"{framework}_{config.data.participant}_S{config.data.session}.log"
    )
    log = setup_logging(
        f"{framework}_{config.data.participant}_{config.data.session}", log_path
    )
    log.info("Config: %s", config_path.name)

    dirs_to_create = [config.results.save_dir, str(config.results.log_dir)]
    setups_dir = getattr(config.results, "setups_dir", None)
    if setups_dir:
        exp_type = config.experiment.type
        dirs_to_create += [
            f"{setups_dir}/{exp_type}/both",
            f"{setups_dir}/{exp_type}/on",
            f"{setups_dir}/{exp_type}/off",
        ]
    for d in dirs_to_create:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    phases_tuple = tuple(p.strip() for p in phases.split(",")) if phases else None
    Pipeline(config, log, phases=phases_tuple, dbs=dbs).run()


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline runner. Framework declared via 'framework.name' in YAML.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Run YAML with a 'framework.name' key.",
    )
    parser.add_argument(
        "--phases",
        default=None,
        help="Comma-separated phase override, e.g. 'train' or 'predictions,forecasts'. Valid: train, predictions, forecasts, classification.",
    )
    parser.add_argument(
        "--dbs",
        default=None,
        choices=["both", "on", "off"],
        help="Train a single DBS condition only. Overrides model_dbs_state in YAML.",
    )
    args = parser.parse_args()

    try:
        run(args.config, phases=args.phases, dbs=args.dbs)
    except ValueError as e:
        print(f"[error] {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
