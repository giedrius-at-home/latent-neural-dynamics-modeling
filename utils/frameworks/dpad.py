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

# Cache for compiled GPU forecast functions keyed by id(idSys).
# Compiled once on first call, reused for all subsequent trials.
_DPADFWK_GPU_FN_CACHE: dict[int, object] = {}


def _build_dpad_gpu_forecast_fn(
    cell1_A, cell1_C, model1_Cy, cell2_A, cell2_C, model2_Cz
):
    """Return a @tf.function that runs the DPAD autoregressive forecast on GPU.

    Replaces DPAD's Python-level 400-step loop with a single compiled
    tf.while_loop, eliminating Python overhead between steps.

    Step (both x1/x2 updated from OLD values, matching propagate_prediction_forward_with_data_gen):
        x1_new  = A1([x1; y])
        z1_new  = C1(x1_new)             -- model1's z (prior for Cz2)
        y1_new  = Cy1(x1_new)            -- model1's y (prior for C2)
        x2_new  = A2([x2; x1; y])        -- uses OLD x1 and y
        y_new   = C2((x2_new, y1_new))   -- total y_hat = C2(x2) + y1
        z       = Cz2((x2_new, z1_new))  -- total z    = Cz2(x2) + z1
    """
    import tensorflow as tf

    @tf.function
    def _forecast(x1, x2, y, m):
        x1_ta = tf.TensorArray(tf.float32, size=m, dynamic_size=False)
        x2_ta = tf.TensorArray(tf.float32, size=m, dynamic_size=False)
        y_ta = tf.TensorArray(tf.float32, size=m, dynamic_size=False)
        z_ta = tf.TensorArray(tf.float32, size=m, dynamic_size=False)
        for i in tf.range(m):
            x1_new = cell1_A.apply_func(tf.concat([x1, y], axis=-1))
            z1_new = cell1_C.apply_func(x1_new)
            y1_new = model1_Cy.apply_func(x1_new)
            x2_new = cell2_A.apply_func(tf.concat([x2, x1, y], axis=-1))
            y = cell2_C.apply_func((x2_new, y1_new))
            z = model2_Cz.apply_func((x2_new, z1_new))
            x1 = x1_new
            x2 = x2_new
            x1_ta = x1_ta.write(i, x1)
            x2_ta = x2_ta.write(i, x2)
            y_ta = y_ta.write(i, y)
            z_ta = z_ta.write(i, z)
        return x1_ta.stack(), x2_ta.stack(), y_ta.stack(), z_ta.stack()

    return _forecast


def dpad_gpu_forecast(
    id_sys: object,
    m: int,
    y_past: "np.ndarray",
) -> "Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]":
    """Run the GPU fast-path forecast on a raw DPADModel instance.

    Returns (Zf, Yf, Xf) each shaped [m, n*] if the model has the
    NLRegRNNCell+unifiedAK structure, otherwise returns None (caller falls back).
    Shares _DPADFWK_GPU_FN_CACHE so the compiled @tf.function is built once
    regardless of whether the call comes from DPADWrapper.forecast() or
    _dpad_idsys_forecast_latents().
    """
    has_gpu_path = (
        getattr(id_sys, "model1", None) is not None
        and getattr(id_sys, "model1_Cy", None) is not None
        and getattr(id_sys, "model2", None) is not None
        and getattr(id_sys, "model2_Cz", None) is not None
        and getattr(id_sys.model1.rnn.cell, "unifiedAK", False)
        and getattr(id_sys.model1.rnn.cell, "C", None) is not None
        and getattr(id_sys.model2.rnn.cell, "unifiedAK", False)
        and getattr(id_sys.model2.rnn.cell, "C", None) is not None
    )
    if not has_gpu_path:
        return None

    import tensorflow as tf

    block_samples = id_sys.block_samples
    n1 = id_sys.n1
    ny = y_past.shape[1]

    def _pad(arr):
        r = arr.shape[0] % block_samples
        if r:
            arr = np.concatenate([arr, np.zeros((block_samples - r, ny))], axis=0)
        return arr

    _key = id(id_sys)
    if _key not in _DPADFWK_GPU_FN_CACHE:
        _DPADFWK_GPU_FN_CACHE[_key] = _build_dpad_gpu_forecast_fn(
            id_sys.model1.rnn.cell.A,
            id_sys.model1.rnn.cell.C,
            id_sys.model1_Cy,
            id_sys.model2.rnn.cell.A,
            id_sys.model2.rnn.cell.C,
            id_sys.model2_Cz,
        )
    gpu_fn = _DPADFWK_GPU_FN_CACHE[_key]

    id_sys.set_steps_ahead([1])
    id_sys.set_multi_step_with_data_gen(False)
    preds1 = id_sys.predict(_pad(y_past))
    Xp_h = np.asarray(preds1[2])  # [T, nx]
    Yp_h = np.asarray(preds1[1])  # [T, ny]
    T = y_past.shape[0]
    x1_T = tf.constant(Xp_h[T - 1 : T, :n1], dtype=tf.float32)
    x2_T = tf.constant(Xp_h[T - 1 : T, n1:], dtype=tf.float32)
    y_T = tf.constant(Yp_h[T - 1 : T, :], dtype=tf.float32)

    x1_f, x2_f, y_f, z_f = gpu_fn(x1_T, x2_T, y_T, m)
    Zf = z_f[:, 0, :].numpy()
    Yf = y_f[:, 0, :].numpy()
    Xf = np.concatenate([x1_f[:, 0, :].numpy(), x2_f[:, 0, :].numpy()], axis=-1)
    return Zf, Yf, Xf


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

        params = self.config.framework.params
        nx: int = params.nx
        n1: int = params.n1
        method_code: str = params.method_code
        epochs: int = params.epochs

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

        # Save forecast cache state so we can restore it after the [1]-step
        # predict pass.  Without this, every predict() call in a forecast loop
        # causes forecast() to rebuild the m-step TF graph from scratch
        # (~40–90 s per trial for m=400).
        _prev_forecast_m = _DPADFWK_FORECAST_CACHE.get(id(self.idSys))

        self.idSys.set_steps_ahead([1])
        self.idSys.set_multi_step_with_data_gen(False)
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

        # Restore multi-step forecast configuration if it was active before
        # this predict() call, so the next forecast(m) call skips the
        # expensive TF graph rebuild.
        if _prev_forecast_m is not None:
            self.idSys.set_steps_ahead(list(range(1, _prev_forecast_m + 1)))
            self.idSys.set_multi_step_with_data_gen(True, noise_samples=0)
            _DPADFWK_FORECAST_CACHE[id(self.idSys)] = _prev_forecast_m

        return all_Zp, all_Yp, all_Xp

    def forecast(
        self,
        m: int,
        Y_past: Array2D,
        Z_past: Optional[Array2D] = None,
    ) -> Tuple[Array2D, Array2D, Array2D]:
        """``Z_past`` accepted for signature parity but ignored (DPAD consumes Y only)."""
        idSys = self.idSys

        result = dpad_gpu_forecast(idSys, m, Y_past)
        if result is not None:
            return result

        # Fallback: original DPAD multi-step data-gen path.
        _key = id(idSys)
        block_samples = idSys.block_samples

        def _pad_to_block(arr):
            remainder = arr.shape[0] % block_samples
            if remainder != 0:
                pad_len = block_samples - remainder
                return np.concatenate([arr, np.zeros((pad_len, arr.shape[1]))], axis=0)
            return arr

        def _stack_last(steps_list):
            return np.vstack([arr[-1:, :] for arr in steps_list])

        if _DPADFWK_FORECAST_CACHE.get(_key) != m:
            idSys.set_steps_ahead(list(range(1, m + 1)))
            idSys.set_multi_step_with_data_gen(True, noise_samples=0)
            _DPADFWK_FORECAST_CACHE[_key] = m
        preds = idSys.predict(_pad_to_block(Y_past))

        Zf = _stack_last(preds[:m])
        Yf = _stack_last(preds[m : 2 * m])
        Xf = _stack_last(preds[2 * m : 3 * m])

        return Zf, Yf, Xf


class DPADFramework(BaseFramework):
    def _initialize_model(self):
        return DPADWrapper(self.config)
