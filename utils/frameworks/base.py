from typing import Any, Dict, List, Optional
from numpy.typing import NDArray
from utils.config import Config
from utils.logger import get_logger
from utils.stats import pearson_r_per_channel
import numpy as np

Array2D = NDArray[np.float64]
TrialList = List[Array2D]


class BaseFramework:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.logger = get_logger()

    def _initialize_model(self):
        raise NotImplementedError

    def _train(self, Y: TrialList, Z: TrialList):
        self.logger.info("Initializing model and starting training.")
        self.model = self._initialize_model()
        self.logger.info(f"Model initialized: {self.model}")
        return self.model.train(Y, Z)

    def _predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        self.logger.info("Running prediction on provided data...")
        return self.model.predict(Y, Z)

    def _forecast(self, m: int, Y_past: Array2D, Z_past: Optional[Array2D] = None):
        self.logger.info(f"Running {m}-step ahead forecast...")
        return self.model.forecast(m, Y_past, Z_past)

    def _evaluate_forecast(
        self,
        Y_list: TrialList,
        Z_list: TrialList,
        margin: float,
    ) -> Dict[str, Any]:
        """Unified multi-step forecast eval loop for all frameworks.

        Per trial: split into ``history_end`` samples of past + ``m`` samples of
        future, call ``self.model.forecast(m, Y_past, Z_past)``, compare forecast
        ``(Yf, Zf)`` against true future. Returns per-trial lists of arrays
        plus aggregate Pearson means.
        """
        self.logger.info("Starting forecast validation...")
        m_seconds = self.config.model.forecast.m
        sampling_freq = self.config.data.sampling_frequency
        m = int(m_seconds * sampling_freq)
        history_end = int(margin * sampling_freq)

        results: Dict[str, Any] = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],
            "Z_future_true": [],
            "Z_future_pred": [],
            "Z_concat_for_plot": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
            "pearson_per_channel_Z": [],
        }

        for idx, Y in enumerate(Y_list):
            Z = Z_list[idx]
            Y_past = Y[:history_end]
            Y_future_true = Y[history_end : history_end + m]
            Z_past = Z[:history_end]
            Z_future_true = Z[history_end : history_end + m]

            Zf, Yf, Xf = self.model.forecast(m, Y_past, Z_past=Z_past)

            Y_concat = np.concatenate([Y_past, Yf], axis=0)
            Z_concat = np.concatenate([Z_past, Zf], axis=0)

            r_list, _ = pearson_r_per_channel([Y_future_true], [Yf])
            r_list = (
                r_list[0] if isinstance(r_list, list) and len(r_list) > 0 else r_list
            )
            r_list_Z, _ = pearson_r_per_channel([Z_future_true], [Zf])
            r_list_Z = (
                r_list_Z[0]
                if isinstance(r_list_Z, list) and len(r_list_Z) > 0
                else r_list_Z
            )

            results["Y_future_true"].append(Y_future_true.tolist())
            results["Y_future_pred"].append(Yf.tolist())
            results["Y_concat_for_plot"].append(Y_concat.tolist())
            results["Z_future_true"].append(Z_future_true.tolist())
            results["Z_future_pred"].append(Zf.tolist())
            results["Z_concat_for_plot"].append(Z_concat.tolist())
            results["X_future_pred"].append(Xf.tolist())
            results["pearson_per_channel"].append(r_list)
            results["pearson_per_channel_Z"].append(r_list_Z)

        flat_r = [
            v
            for r in results["pearson_per_channel"]
            if r
            for v in r
            if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean"] = float(np.mean(flat_r)) if flat_r else np.nan
        flat_r_Z = [
            v
            for r in results["pearson_per_channel_Z"]
            if r
            for v in r
            if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean_Z"] = (
            float(np.mean(flat_r_Z)) if flat_r_Z else np.nan
        )

        return results


class BaseWrapper:
    """Shared surface for PSID/DPAD/VARMA wrappers.

    Subclasses implement: ``train``, ``predict``, ``forecast``,
    ``load_from_file``. Training writes the model to disk; testing loads it
    and calls ``predict`` / ``forecast`` via ``BaseFramework._predict`` /
    ``_evaluate_forecast``. No separate predict-side validation method —
    ``predict`` is the one-step pass, ``_evaluate_forecast`` is multi-step.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.idSys = None

    def train(self, Y: TrialList, Z: TrialList):
        raise NotImplementedError

    def predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        raise NotImplementedError

    def forecast(self, m: int, Y_past: Array2D, Z_past: Optional[Array2D] = None):
        raise NotImplementedError

    def load_from_file(self, model_path: str):
        raise NotImplementedError
