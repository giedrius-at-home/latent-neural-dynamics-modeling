import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
from typing import Tuple, Dict, List, Optional


def compute_rdm(data: np.ndarray, metric: str = "correlation") -> np.ndarray:
    if metric == "correlation" and (data.ndim == 1 or data.shape[1] == 1):
        metric = "euclidean"

    distances = pdist(data, metric=metric)
    rdm = squareform(distances)

    return rdm


def rsa_correlation(rdm1: np.ndarray, rdm2: np.ndarray) -> Tuple[float, float]:
    if rdm1.shape != rdm2.shape:
        raise ValueError(f"RDM shapes must match: {rdm1.shape} vs {rdm2.shape}")

    if rdm1.shape[0] != rdm1.shape[1]:
        raise ValueError(f"RDMs must be square: {rdm1.shape}")

    triu_idx = np.triu_indices(rdm1.shape[0], k=1)
    vec1 = rdm1[triu_idx]
    vec2 = rdm2[triu_idx]

    r, p = scipy.stats.pearsonr(vec1, vec2)

    return r, p


def rsa_multiple_comparisons(
    rdms: Dict[str, np.ndarray],
    alpha: float = 0.05,
    correction_method: str = "bonferroni",
) -> List[Dict]:

    rdm_names = list(rdms.keys())
    n_rdms = len(rdm_names)

    comparisons = []
    p_values = []

    for i in range(n_rdms):
        for j in range(i + 1, n_rdms):
            name1, name2 = rdm_names[i], rdm_names[j]
            rdm1, rdm2 = rdms[name1], rdms[name2]

            r, p = rsa_correlation(rdm1, rdm2)

            comparisons.append(
                {
                    "comparison": f"{name1} vs {name2}",
                    "rdm1": name1,
                    "rdm2": name2,
                    "correlation": r,
                    "p_original": p,
                }
            )
            p_values.append(p)

    reject, p_corrected, _, _ = multipletests(
        p_values, alpha=alpha, method=correction_method
    )

    for i, comp in enumerate(comparisons):
        comp["p_corrected"] = p_corrected[i]
        comp["significant"] = reject[i]
        comp["alpha"] = alpha
        comp["correction_method"] = correction_method

    return comparisons


def compute_rdm_from_trials(
    trial_data: List[np.ndarray], aggregation: str = "mean", metric: str = "correlation"
) -> np.ndarray:
    if aggregation == "mean":
        trial_reps = np.array([np.mean(trial, axis=0) for trial in trial_data])
    elif aggregation == "median":
        trial_reps = np.array([np.median(trial, axis=0) for trial in trial_data])
    elif aggregation == "last":
        trial_reps = np.array([trial[-1] for trial in trial_data])
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return compute_rdm(trial_reps, metric=metric)
