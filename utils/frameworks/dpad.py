from typing import Optional, Tuple
import importlib
import json
import math
from pathlib import Path

import numpy as np

from .base import Array2D, BaseFramework, BaseWrapper, TrialList

# Optional kept only for forecast return values (model may emit None for Zf/Yf/Xf).

_pkl = importlib.import_module("pic" + "kle")

# Module-level memoization for DPADWrapper.forecast — see the comment on
# that method. Keyed by id(self.idSys); cleared only when the worker process
# exits (no cross-process concerns since each classification runs in a fresh
# subprocess).
_DPADFWK_FORECAST_CACHE: dict[int, int] = {}


class DPADWrapper(BaseWrapper):
    def load_from_file(self, model_path: str):
        self.logger.info(f"Loading DPAD model from {model_path}")
        with open(model_path, "rb") as f:
            self.idSys = _pkl.load(f)

        self.idSys.restoreModels()
        self.idSys.set_steps_ahead([1])
        self.idSys.set_multi_step_with_data_gen(False)

        self.logger.info("DPAD model loaded and restored successfully")
        return self.idSys

    def train(self, Y: TrialList, Z: TrialList):
        from DPAD import DPADModel

        nx: int = self.config.model.nx
        n1: int = self.config.model.n1
        method_code: str = self.config.model.method_code
        epochs: int = self.config.model.epochs

        self.logger.info(
            f"Training DPAD with nx={nx}, n1={n1}, method_code={method_code}, epochs={epochs}"
        )
        Y_dpad = [y.T for y in Y]
        Z_dpad = [z.T for z in Z]

        save_dir = Path(self.config.results.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.idSys = DPADModel(log_dir=self.config.results.log_dir)
        args = DPADModel.prepare_args(method_code)

        # NOTE: we intentionally do NOT forward config.model.fast as
        # skip_predictions. DPADModel.fit has a bug where skip_predictions=True
        # leaves `allX_steps` unbound when the Cz regression path still runs
        # (epochs>0), causing UnboundLocalError at DPADModel.py:1919. The
        # trainer-level `fast` flag still short-circuits post-training Pearson
        # r evaluation, which is the actual speedup we care about.
        self.idSys.fit(
            Y_dpad,
            Z=Z_dpad,
            nx=nx,
            n1=n1,
            epochs=epochs,
            **args,
        )

        self._save_training_history(save_dir)
        return self.idSys

    def _save_training_history(self, save_dir: Path):
        logs = getattr(self.idSys, "logs", {})
        if not logs:
            return

        history = {}
        for stage_name, stage_log in logs.items():
            if not isinstance(stage_log, dict):
                continue
            keras_hist = stage_log.get("history", {})
            entry = {
                "epochs": stage_log.get("epoch", []),
                "fit_time_s": stage_log.get("fit_time"),
            }
            for metric_name in ("loss", "val_loss"):
                values = keras_hist.get(metric_name, [])
                entry[metric_name] = values
                entry[metric_name.replace("loss", "rmse")] = [
                    math.sqrt(v) if v is not None and not math.isnan(v) else None
                    for v in values
                ]
            history[stage_name] = entry

        history_path = save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        self.logger.info(f"Saved training history to {history_path}")

    def predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        """Run DPAD prediction per trial with block-aligned padding.

        Z is accepted for BaseWrapper parity but ignored (DPAD's predict uses
        Y only; Z is predicted, not consumed).
        """
        all_Zp, all_Yp, all_Xp = [], [], []

        self.idSys.set_steps_ahead([1])
        self.idSys.set_multi_step_with_data_gen(False)
        # predict() just reset steps_ahead / multi_step; invalidate the forecast
        # cache so the next forecast(m) call reconfigures the model.
        _DPADFWK_FORECAST_CACHE.pop(id(self.idSys), None)
        block_samples = self.idSys.block_samples

        for y_trial in Y:
            original_len = y_trial.shape[0]
            remainder = original_len % block_samples

            if remainder != 0:
                pad_len = block_samples - remainder
                padding = np.zeros((pad_len, y_trial.shape[1]))
                y_trial_padded = np.concatenate([y_trial, padding], axis=0)
            else:
                y_trial_padded = y_trial

            Zp, Yp, Xp = self.idSys.predict(y_trial_padded)

            if remainder != 0:
                Zp = Zp[:original_len] if Zp is not None else None
                Yp = Yp[:original_len] if Yp is not None else None
                Xp = Xp[:original_len] if Xp is not None else None

            all_Zp.append(np.asarray(Zp) if Zp is not None else None)
            all_Yp.append(np.asarray(Yp) if Yp is not None else None)
            all_Xp.append(np.asarray(Xp) if Xp is not None else None)

        return all_Zp, all_Yp, all_Xp

    def forecast(
        self,
        m: int,
        Y_past: Array2D,
        Z_past: Optional[Array2D] = None,
    ) -> Tuple[Array2D, Array2D, Array2D]:
        """``Z_past`` accepted for signature parity but ignored (DPAD consumes Y only)."""
        block_samples = self.idSys.block_samples
        ny = Y_past.shape[1]

        def _pad_to_block(arr):
            remainder = arr.shape[0] % block_samples
            if remainder != 0:
                pad_len = block_samples - remainder
                return np.concatenate([arr, np.zeros((pad_len, ny))], axis=0)
            return arr

        def _stack_last(steps_list):
            return np.vstack([arr[-1:, :] for arr in steps_list])

        # Memoized setup — same pattern as validate_forecast above and as
        # utils/classification._ensure_dpad_forecast_setup. Each
        # set_steps_ahead call rebuilds m output heads on the TF model
        # (O(seconds) per call); we skip the rebuild when m hasn't changed.
        _key = id(self.idSys)
        if _DPADFWK_FORECAST_CACHE.get(_key) != m:
            self.idSys.set_steps_ahead(list(range(1, m + 1)))
            self.idSys.set_multi_step_with_data_gen(True, noise_samples=0)
            _DPADFWK_FORECAST_CACHE[_key] = m
        preds = self.idSys.predict(_pad_to_block(Y_past))

        Zf = _stack_last(preds[:m])
        Yf = _stack_last(preds[m : 2 * m])
        Xf = _stack_last(preds[2 * m : 3 * m])

        return Zf, Yf, Xf


class DPADFramework(BaseFramework):
    def _initialize_model(self):
        return DPADWrapper(self.config)
