"""Classification package — split from the old 1.1k-line utils/classification.py.

Layout:
- ``splits.py``    : ``ChronoGroupsSplit`` (block-aware K-fold CV splitter).
- ``data.py``      : trial-level prep (epoching, feature scoping, flipped
                      counterfactual latent generation) and on-disk loaders.
- ``pipeline.py``  : sklearn pipeline + fixed-param block-aware CV loop
                      (``run_cv``; replaces the old ``run_grid_search_cv``).
- ``evaluation.py``: test-set metrics, group-level label permutation test, and
                      the LDA orchestration functions consumed by
                      ``classification/compute.py``.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    _load_framework_for_forecast,
    load_all_splits,
    load_precomputed_results,
    prepare_epoched_data,
    prepare_ground_truth_eval_data,
)
from .evaluation import (
    apply_lda_permutation_to_results,
    evaluate_on_test_set,
    run_classification_pipeline,
    run_lda_fit_and_test_only,
    run_permutation_test,
)
from .pipeline import create_pipeline, run_cv
from .splits import ChronoGroupsSplit

__all__ = [
    "ChronoGroupsSplit",
    "Pipeline",
    "StandardScaler",
    "apply_lda_permutation_to_results",
    "create_pipeline",
    "evaluate_on_test_set",
    "load_all_splits",
    "load_precomputed_results",
    "prepare_epoched_data",
    "prepare_ground_truth_eval_data",
    "run_classification_pipeline",
    "run_cv",
    "run_lda_fit_and_test_only",
    "run_permutation_test",
]
