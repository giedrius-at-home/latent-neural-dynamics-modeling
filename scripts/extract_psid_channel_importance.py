#!/usr/bin/env python3
"""
Extract channel importance from a trained PSID model using Cy, Cz, and A eigenvalues.

Ranking methods:
  1. ||Cy[i, :]|| — how much latent space captures each neural channel
  2. Eigenvalue-weighted: ||Cy[i, :] @ diag(|lambda|)|| — emphasizes persistent dynamics
  3. Per-subspace: separate rankings for x1 (behaviorally-relevant) and x2 (remaining)

Usage:
    python scripts/extract_psid_channel_importance.py \
        --model results/psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band/model_20260408_185522.pkl \
        --n1 6 --top 20
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def extract_channel_importance(
    model_path: str | Path,
    n1: int,
    channel_names: list[str] | None = None,
    sampling_freq: float = 200.0,
) -> dict:
    """Extract channel importance from a trained PSID model.

    Returns dict with:
        - channel_importance: array of importance scores per channel
        - channel_importance_weighted: eigenvalue-weighted scores
        - channel_importance_x1: importance from behaviorally-relevant subspace
        - channel_importance_x2: importance from remaining subspace
        - eigenvalues: A matrix eigenvalues
        - rankings: sorted channel indices (most important first)
    """
    with open(model_path, "rb") as f:
        idSys = pickle.load(f)

    A = np.array(idSys.A)
    Cy = np.array(idSys.C) if hasattr(idSys, "C") else np.array(idSys.Cy)
    Cz = np.array(idSys.Cz) if hasattr(idSys, "Cz") else None

    nx = A.shape[0]
    ny = Cy.shape[0]

    # Eigenvalues of A
    eigenvalues = np.linalg.eigvals(A)
    magnitudes = np.abs(eigenvalues)

    # 1. Raw importance: ||Cy[i, :]|| per channel
    raw_importance = np.linalg.norm(Cy, axis=1)

    # 2. Eigenvalue-weighted: weight columns of Cy by eigenvalue magnitudes
    # Channels that load on slow/persistent modes get higher scores
    weighted_Cy = Cy @ np.diag(magnitudes)
    weighted_importance = np.linalg.norm(weighted_Cy, axis=1)

    # 3. Per-subspace importance
    Cy_x1 = Cy[:, :n1]       # behaviorally-relevant subspace
    Cy_x2 = Cy[:, n1:]       # remaining subspace
    importance_x1 = np.linalg.norm(Cy_x1, axis=1)
    importance_x2 = np.linalg.norm(Cy_x2, axis=1)

    # 4. Eigenvalue analysis for A11 and A22
    A11 = A[:n1, :n1]
    A22 = A[n1:, n1:]
    eig_x1 = np.linalg.eigvals(A11)
    eig_x2 = np.linalg.eigvals(A22)

    # Rankings (descending importance)
    rank_raw = np.argsort(raw_importance)[::-1]
    rank_weighted = np.argsort(weighted_importance)[::-1]
    rank_x1 = np.argsort(importance_x1)[::-1]
    rank_x2 = np.argsort(importance_x2)[::-1]

    result = {
        "raw_importance": raw_importance,
        "weighted_importance": weighted_importance,
        "importance_x1": importance_x1,
        "importance_x2": importance_x2,
        "rank_raw": rank_raw,
        "rank_weighted": rank_weighted,
        "rank_x1": rank_x1,
        "rank_x2": rank_x2,
        "eigenvalues": eigenvalues,
        "eig_x1": eig_x1,
        "eig_x2": eig_x2,
        "A": A,
        "Cy": Cy,
        "Cz": Cz,
        "nx": nx,
        "n1": n1,
        "ny": ny,
    }

    if channel_names is not None:
        result["channel_names"] = channel_names

    return result


def get_top_channels(
    importance_result: dict,
    top_n: int,
    method: str = "weighted",
    channel_names: list[str] | None = None,
) -> list[str] | list[int]:
    """Return top-N channel names (or indices) by importance."""
    rank_key = f"rank_{method}"
    ranking = importance_result[rank_key]
    top_indices = ranking[:top_n].tolist()

    names = channel_names or importance_result.get("channel_names")
    if names:
        return [names[i] for i in top_indices]
    return top_indices


def compute_behavioral_relevance(
    importance_result: dict,
    channel_names: list[str] | None = None,
) -> dict:
    """Compute combined Cy→Cz behavioral relevance per channel.

    For each channel i:  behavioral_relevance[i] = ||Cz @ Cy[i, :].T||
    This traces the full path: neural channel → latent states → behavior.

    Also computes x1-only relevance (through behaviorally-relevant subspace only).

    Returns dict with:
        - behavioral_relevance: array (ny,)
        - behavioral_relevance_x1: array (ny,) via x1 subspace only
        - rank_behavioral: sorted indices (descending)
        - rank_behavioral_x1: sorted indices (descending)
    """
    Cy = importance_result["Cy"]
    Cz = importance_result["Cz"]
    n1 = importance_result["n1"]

    if Cz is None:
        return None

    # Full path: channel → all latent states → behavior
    behavioral_relevance = np.array(
        [np.linalg.norm(Cz @ Cy[i, :]) for i in range(Cy.shape[0])]
    )

    # x1-only path: channel → behaviorally-relevant subspace → behavior
    Cz_x1 = Cz[:, :n1]
    Cy_x1 = Cy[:, :n1]
    behavioral_relevance_x1 = np.array(
        [np.linalg.norm(Cz_x1 @ Cy_x1[i, :]) for i in range(Cy.shape[0])]
    )

    rank_br = np.argsort(behavioral_relevance)[::-1]
    rank_br_x1 = np.argsort(behavioral_relevance_x1)[::-1]

    result = {
        "behavioral_relevance": behavioral_relevance,
        "behavioral_relevance_x1": behavioral_relevance_x1,
        "rank_behavioral": rank_br,
        "rank_behavioral_x1": rank_br_x1,
    }

    names = channel_names or importance_result.get("channel_names")
    if names:
        result["channel_names"] = names

    return result


def get_top_behavioral_and_neural(
    model_path: str | Path,
    n1: int,
    channel_names: list[str],
    top_n: int = 5,
) -> dict:
    """Return the top-N channels for behavioral relevance and neural dynamics.

    Behavioral: ranked by ||Cz @ Cy[i,:]|| (full Cy→Cz path)
    Neural:     ranked by eigenvalue-weighted ||Cy[i,:] @ diag(|λ|)||

    Returns dict with:
        - top_behavioral: list of channel names
        - top_neural: list of channel names
        - top_behavioral_scores: corresponding scores
        - top_neural_scores: corresponding scores
        - all results from extract_channel_importance and compute_behavioral_relevance
    """
    imp = extract_channel_importance(model_path, n1, channel_names)
    br = compute_behavioral_relevance(imp, channel_names)

    # Top behavioral channels (Cy→Cz combined)
    rank_beh = br["rank_behavioral"]
    top_beh_idx = rank_beh[:top_n].tolist()
    top_beh_names = [channel_names[i] for i in top_beh_idx]
    top_beh_scores = [float(br["behavioral_relevance"][i]) for i in top_beh_idx]

    # Top neural channels (eigenvalue-weighted Cy)
    rank_neural = imp["rank_weighted"]
    top_neural_idx = rank_neural[:top_n].tolist()
    top_neural_names = [channel_names[i] for i in top_neural_idx]
    top_neural_scores = [float(imp["weighted_importance"][i]) for i in top_neural_idx]

    return {
        "top_behavioral": top_beh_names,
        "top_neural": top_neural_names,
        "top_behavioral_scores": top_beh_scores,
        "top_neural_scores": top_neural_scores,
        "importance": imp,
        "behavioral_relevance": br,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model pkl")
    parser.add_argument("--n1", type=int, required=True, help="Number of behaviorally-relevant states")
    parser.add_argument("--top", type=int, default=20, help="Number of top channels to show")
    parser.add_argument("--channels-json", type=str, default=None,
                        help="JSON file with channel names list")
    args = parser.parse_args()

    channel_names = None
    if args.channels_json:
        with open(args.channels_json) as f:
            channel_names = json.load(f)

    result = extract_channel_importance(args.model, args.n1, channel_names)

    print(f"\nPSID Model: nx={result['nx']}, n1={result['n1']}, ny={result['ny']}")
    print(f"A spectral radius: {np.max(np.abs(result['eigenvalues'])):.4f}")
    print(f"A11 spectral radius: {np.max(np.abs(result['eig_x1'])):.4f}")
    print(f"A22 spectral radius: {np.max(np.abs(result['eig_x2'])):.4f}")

    for method in ("raw", "weighted", "x1", "x2"):
        rank = result[f"rank_{method}"]
        imp_key = {
            "raw": "raw_importance",
            "weighted": "weighted_importance",
            "x1": "importance_x1",
            "x2": "importance_x2",
        }[method]
        scores = result[imp_key]

        print(f"\n--- Top {args.top} channels ({method}) ---")
        for i, idx in enumerate(rank[:args.top]):
            name = channel_names[idx] if channel_names else f"ch_{idx}"
            print(f"  {i+1:3d}. {name:40s}  score={scores[idx]:.6f}")


if __name__ == "__main__":
    main()
