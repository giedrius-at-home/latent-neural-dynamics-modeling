# Cell 1: Imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc, balanced_accuracy_score
from mne.decoding import SPoC

from utils.classification import prepare_epoched_data, get_classifier
from dashboard.subtabs import load_precomputed_results

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)

# Cell 2: Config
PROJECT_ROOT = Path.cwd().parent
RESULTS_ROOT = PROJECT_ROOT / "results"

VARIANT = "dpad_behavioral_PDI4_S3"
RUN_TS = "20251208_232439"

FEATURE_SOURCE = "Xp"
EPOCH_LENGTH_SEC = 9
EPOCH_OVERLAP = 0
N_SPLITS = 2
CLASSIFIER = "LDA"
SPOC_N_COMPONENTS = 4

# Cell 3: Load data
variant_dir = RESULTS_ROOT / VARIANT

train_res = load_precomputed_results(variant_dir, RUN_TS, "train")
val_res = load_precomputed_results(variant_dir, RUN_TS, "val")

trainval_list = [r for r in [train_res, val_res] if r is not None]

X, y, meta = prepare_epoched_data(
    trainval_list,
    feature_source=FEATURE_SOURCE,
    epoch_length_sec=EPOCH_LENGTH_SEC,
    overlap=EPOCH_OVERLAP,
)

print(f"X shape: {X.shape}")
print(f"Total epochs: {len(X)}")
print(f"DBS ON: {np.sum(y == 1)} | DBS OFF: {np.sum(y == 0)}")

# Cell 4: CV with SPoC
if CLASSIFIER == "LDA":
    clf = get_classifier("LDA", {"solver": "lsqr", "shrinkage": "auto"})
else:
    clf = get_classifier(
        "Logistic Regression", {"C": 1.0, "penalty": "l2", "solver": "liblinear"}
    )

pipeline = Pipeline(
    [
        ("spoc", SPoC(n_components=SPOC_N_COMPONENTS, reg="ledoit_wolf", log=True)),
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ]
)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

fold_results = []
all_y_true = []
all_y_pred = []
all_y_proba = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    bal_acc = balanced_accuracy_score(y_val, y_pred)

    fold_results.append(
        {
            "fold": fold_idx,
            "bal_acc": bal_acc,
            "n_train": len(y_train),
            "n_val": len(y_val),
        }
    )

    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

    print(f"Fold {fold_idx}: Bal Acc = {bal_acc:.4f}")

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)

mean_bal_acc = np.mean([r["bal_acc"] for r in fold_results])
print(f"\nMean Balanced Accuracy: {mean_bal_acc:.4f}")

# Cell 5: Performance plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

fold_nums = [r["fold"] for r in fold_results]
bal_accs = [r["bal_acc"] for r in fold_results]
axes[0].plot(fold_nums, bal_accs, marker="o", linewidth=2, markersize=8)
axes[0].axhline(
    mean_bal_acc, color="r", linestyle="--", label=f"Mean: {mean_bal_acc:.3f}"
)
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Balanced Accuracy")
axes[0].set_title("Per-Fold Performance")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

cm = confusion_matrix(all_y_true, all_y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1],
    xticklabels=["OFF", "ON"],
    yticklabels=["OFF", "ON"],
)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].set_title("Confusion Matrix (All Folds)")

fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
roc_auc = auc(fpr, tpr)
axes[2].plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
axes[2].plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curve")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 6: SPoC patterns
pipeline.fit(X, y)
spoc = pipeline.named_steps["spoc"]
patterns = spoc.patterns_

n_components = patterns.shape[1]
fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 3))
if n_components == 1:
    axes = [axes]

for i in range(n_components):
    axes[i].plot(patterns[:, i], linewidth=2)
    axes[i].set_title(f"SPoC Pattern {i+1}")
    axes[i].set_xlabel("Channel")
    axes[i].set_ylabel("Weight")
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(0, color="k", linestyle="--", linewidth=0.8)

plt.tight_layout()
plt.show()

# Cell 7: Classifier weights
classifier = pipeline.named_steps["classifier"]

if hasattr(classifier, "coef_"):
    weights = classifier.coef_[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.bar(range(len(weights)), weights, color="steelblue")
    ax.set_xlabel("SPoC Component")
    ax.set_ylabel("Classifier Weight")
    ax.set_title(f"{CLASSIFIER} Weights on SPoC Components")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f"Comp {i+1}" for i in range(len(weights))])

    plt.tight_layout()
    plt.show()
else:
    print(f"{CLASSIFIER} does not have interpretable weights")
