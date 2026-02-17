from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import pickle


def compute_psid_feature_importance(
    idSys, input_channels: List[str]
) -> Dict[str, float]:
    if not hasattr(idSys, "K") or idSys.K is None:
        raise ValueError("Model does not have Kalman gain matrix K")

    K = np.array(idSys.K)
    n1 = getattr(idSys, "n1", K.shape[0])

    K_behavioral = K[:n1, :]

    importance_scores = np.linalg.norm(K_behavioral, axis=0)

    if len(input_channels) != importance_scores.shape[0]:
        raise ValueError(
            f"Channel count mismatch: {len(input_channels)} channels vs "
            f"{importance_scores.shape[0]} K columns"
        )

    return {ch: float(score) for ch, score in zip(input_channels, importance_scores)}


def rank_features(importance_dict: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)


def load_model_and_compute_importance(
    model_path: Path, input_channels: List[str]
) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    with open(model_path, "rb") as f:
        idSys = pickle.load(f)

    importance = compute_psid_feature_importance(idSys, input_channels)
    ranked = rank_features(importance)

    return importance, ranked


def normalize_importance(importance_dict: Dict[str, float]) -> Dict[str, float]:
    values = np.array(list(importance_dict.values()))
    max_val = np.max(values) if len(values) > 0 else 1.0
    if max_val == 0:
        max_val = 1.0
    return {k: v / max_val for k, v in importance_dict.items()}
