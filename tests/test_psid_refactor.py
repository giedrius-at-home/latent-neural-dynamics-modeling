"""Verification tests for PSIDWrapper after the PSID vendor-fork refactor.

Run with:
    pytest tests/test_psid_refactor.py -v
or:
    python tests/test_psid_refactor.py
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Sanity: we must be using upstream PSID, not the vendored fork.
import PSID as _psid_mod  # noqa: E402
_psid_path = str(Path(_psid_mod.__file__).resolve())
assert "site-packages" in _psid_path, (
    f"Expected upstream PSID from site-packages, got: {_psid_path}. "
    "Rename the vendored ./PSID/ directory before running these tests."
)

from PSID.PSID import PSID as upstream_PSID  # noqa: E402
from utils.frameworks import PSIDWrapper  # noqa: E402


def _canned_data(seed=0, T=5000, ny=6, nz=2, nx=3):
    """Generate stable synthetic LSSM data (SNR tuned for DARE to solve).

    Uses a rotation-based A so |eig(A)| is a tight, well-conditioned bound.
    """
    rng = np.random.default_rng(seed)
    # Block-diagonal A: one rotation block + one decay scalar
    theta = 0.3
    rot = 0.9 * np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
    A_true = np.block([
        [rot, np.zeros((2, nx - 2))],
        [np.zeros((nx - 2, 2)), 0.7 * np.eye(nx - 2)],
    ])
    C_true = rng.normal(size=(ny, nx))
    Cz_true = rng.normal(size=(nz, nx))
    x = np.zeros(nx)
    Y = np.zeros((T, ny))
    Z = np.zeros((T, nz))
    for t in range(T):
        x = A_true @ x + 0.3 * rng.normal(size=nx)
        Y[t] = C_true @ x + 0.15 * rng.normal(size=ny)
        Z[t] = Cz_true @ x + 0.1 * rng.normal(size=nz)
    return Y, Z


def _identifying_matrices(idSys, names=None):
    """Stable attributes expected to match between refactor and upstream."""
    if names is None:
        names = ["A", "C", "Cz", "Q", "R", "S", "Kf", "Pp", "innovCov"]
    out = {}
    for name in names:
        val = getattr(idSys, name, None)
        out[name] = None if val is None else np.asarray(val)
    return out


def _assert_all_close(d_ref, d_test, atol=1e-12, label=""):
    for k in d_ref:
        a = d_ref[k]
        b = d_test[k]
        assert (a is None) == (b is None), f"{label} {k}: None mismatch"
        if a is None:
            continue
        assert a.shape == b.shape, f"{label} {k}: shape {a.shape} vs {b.shape}"
        # Treat matching NaN positions as equal — when DARE fails, both
        # upstream and the refactor propagate NaN to the same entries.
        nan_a = np.isnan(a)
        nan_b = np.isnan(b)
        assert np.array_equal(nan_a, nan_b), f"{label} {k}: NaN pattern differs"
        mask = ~nan_a
        if not mask.any():
            continue
        diff = float(np.max(np.abs(a[mask] - b[mask])))
        assert diff <= atol, f"{label} {k}: max|diff|={diff:g} > {atol:g}"


def test_parity_upstream_with_mods_disabled():
    """Test #1: _psid_identify with additions disabled == upstream PSID().

    Compares only pre-DARE matrices (A, C, Cz, Q, R, S) — these are all the
    identification outputs of the PSID paper equations. DARE-derived fields
    (Pp, Kf, innovCov) are expected to differ because the refactor includes
    a scipy >= 1.16 DARE bug workaround (see test_dare_fix).
    """
    Y, Z = _canned_data(seed=1)
    nx, n1, i = 3, 1, 15

    # Disable fit_Cz_via_KF so Cz comes from raw SVD projection (no Kalman
    # filter dependence). Otherwise upstream's Cz would be computed from
    # non-steady-state Kalman states (because DARE failed in upstream), while
    # refactor's Cz would be computed from steady-state states (because the
    # refactor fixes DARE), and the two would disagree.
    s_ref = upstream_PSID(Y, Z, nx=nx, n1=n1, i=i, fit_Cz_via_KF=False)
    s_test = PSIDWrapper._psid_identify(
        Y, Z, nx=nx, n1=n1, i=i,
        max_eigenvalue=1.0,       # disables damping
        backward_kalman=False,    # upstream behavior
        fit_Cz_via_KF=False,
    )

    pre_dare_fields = ["A", "C", "Cz", "Q", "R", "S"]
    ref = _identifying_matrices(s_ref, names=pre_dare_fields)
    test = _identifying_matrices(s_test, names=pre_dare_fields)
    _assert_all_close(ref, test, atol=1e-12, label="parity(upstream)")
    print("  [ok] pre-DARE parity vs upstream (A, C, Cz-from-SVD, Q, R, S)")


def test_dare_fix_produces_finite_pp():
    """Test #1b: refactor's DARE workaround yields finite Pp / Kf / innovCov.

    On scipy >= 1.16, upstream PSID() returns NaN for Pp/Kf/innovCov because
    solve_discrete_are(... s=S) raises "Matrix e should be square". The
    refactor detects the NaN fallback and re-solves DARE with e=None.
    """
    Y, Z = _canned_data(seed=1)
    nx, n1, i = 3, 1, 15
    s = PSIDWrapper._psid_identify(
        Y, Z, nx=nx, n1=n1, i=i, max_eigenvalue=1.0, backward_kalman=False
    )
    assert s.Pp is not None, "Pp not set after _psid_identify"
    assert not np.any(np.isnan(np.asarray(s.Pp))), (
        "Pp still NaN after _refit_dare_if_needed — scipy workaround failed"
    )
    assert not np.any(np.isnan(np.asarray(s.Kf))), "Kf still NaN"
    assert not np.any(np.isnan(np.asarray(s.innovCov))), "innovCov still NaN"

    # Verify Pp satisfies the DARE: Pp = A Pp A.T + Q - (A Pp C.T + S)(C Pp C.T + R)^-1 (A Pp C.T + S).T
    A = np.asarray(s.A); C = np.asarray(s.C); Q = np.asarray(s.Q)
    R = np.asarray(s.R); S = np.asarray(s.S); Pp = np.asarray(s.Pp)
    # Upstream calls solve_discrete_are(A.T, C.T, Q, R, s=S) which solves:
    #   Pp = A Pp A.T - (A Pp C.T + S)(C Pp C.T + R)^-1 (A Pp C.T + S).T + Q
    innov = C @ Pp @ C.T + R
    G = A @ Pp @ C.T + S
    Pp_from_riccati = A @ Pp @ A.T + Q - G @ np.linalg.solve(innov, G.T)
    residual = np.max(np.abs(Pp - Pp_from_riccati))
    assert residual < 1e-8, f"DARE residual too large: {residual:g}"
    print(f"  [ok] Pp finite, DARE residual={residual:g}")


def test_eigenvalue_damping():
    """Test #4a: max_eigenvalue=0.95 clips |eig(A)|."""
    Y, Z = _canned_data(seed=2)
    nx, n1, i = 3, 1, 15
    max_eig = 0.95

    s = PSIDWrapper._psid_identify(
        Y, Z, nx=nx, n1=n1, i=i,
        max_eigenvalue=max_eig,
        backward_kalman=False,
    )
    eigs = np.abs(np.linalg.eigvals(np.asarray(s.A)))
    max_abs_eig = float(np.max(eigs))
    assert max_abs_eig <= max_eig + 1e-10, (
        f"max|eig(A)|={max_abs_eig} > {max_eig}"
    )
    print(f"  [ok] eigenvalue damping: max|eig|={max_abs_eig:.6f} <= {max_eig}")


def test_smoother_cz_refit():
    """Test #3: backward_kalman=True produces Cz that matches externally-computed smoother Cz."""
    from PSID.PSID import projOrth

    Y, Z = _canned_data(seed=3)
    nx, n1, i = 3, 1, 15

    # Refactor path: _psid_identify with backward_kalman=True
    s_bk = PSIDWrapper._psid_identify(
        Y, Z, nx=nx, n1=n1, i=i,
        max_eigenvalue=1.0,
        backward_kalman=True,
    )
    Cz_refactor = np.asarray(s_bk.Cz)

    # Reference path: identify filter-only Cz, then externally re-fit Cz
    # against smoothed states.
    s_ref = PSIDWrapper._psid_identify(
        Y, Z, nx=nx, n1=n1, i=i,
        max_eigenvalue=1.0,
        backward_kalman=False,
        fit_Cz_via_KF=False,   # skip filter-based refit; use raw projOrth Cz
    )
    PSIDWrapper._attach_smoother_params(s_ref)
    # Upstream's LSSM.kalman applies YPrepModel internally, so pass raw Y.
    # Do the Z prep manually since there's no equivalent in kalman.
    Z_prep = s_ref.ZPrepModel.apply(Z, time_first=True)
    _, _, Xs = PSIDWrapper._rts_smooth(s_ref, Y)
    Cz_ref = projOrth(Z_prep.T, Xs.T)[1]

    diff = float(np.max(np.abs(Cz_refactor - Cz_ref)))
    assert diff < 1e-10, f"smoother Cz mismatch: max|diff|={diff:g}"

    # And verify the smoother Cz is actually *different* from the filter Cz
    # (otherwise backward_kalman=True would be a no-op).
    s_filt = PSIDWrapper._psid_identify(
        Y, Z, nx=nx, n1=n1, i=i,
        max_eigenvalue=1.0,
        backward_kalman=False,
    )
    Cz_filt = np.asarray(s_filt.Cz)
    if np.max(np.abs(Cz_filt - Cz_refactor)) < 1e-10:
        print("  [warn] smoother Cz == filter Cz — backward_kalman is a no-op on this data")
    else:
        print(f"  [ok] smoother Cz differs from filter Cz (max|diff|={np.max(np.abs(Cz_filt - Cz_refactor)):g})")
    print(f"  [ok] smoother Cz matches reference (max|diff|={diff:g})")


def test_smoother_params_attached():
    """Test #4b: Pf and Cs are attached after train."""
    Y, Z = _canned_data(seed=4)
    nx, n1, i = 3, 1, 15
    s = PSIDWrapper._psid_identify(
        Y, Z, nx=nx, n1=n1, i=i, max_eigenvalue=1.0, backward_kalman=False
    )
    PSIDWrapper._attach_smoother_params(s)
    assert hasattr(s, "Pf") and s.Pf is not None, "Pf not attached"
    assert hasattr(s, "Cs") and s.Cs is not None, "Cs not attached"
    # Pf = Pp - Kf @ C @ Pp
    Pp = np.asarray(s.Pp)
    Kf = np.asarray(s.Kf)
    C = np.asarray(s.C)
    Pf_expected = Pp - Kf @ C @ Pp
    assert np.allclose(np.asarray(s.Pf), Pf_expected, atol=1e-12), "Pf formula mismatch"
    Cs_expected = Pf_expected @ np.asarray(s.A).T @ np.linalg.pinv(Pp)
    assert np.allclose(np.asarray(s.Cs), Cs_expected, atol=1e-10), "Cs formula mismatch"
    print("  [ok] Pf / Cs attached correctly")


def test_rts_smoother_matches_ref():
    """Test #5: RTS backward pass produces a smoothed trajectory consistent with a manual implementation."""
    Y, Z = _canned_data(seed=5, T=500)
    s = PSIDWrapper._psid_identify(
        Y, Z, nx=4, n1=2, i=10, max_eigenvalue=1.0, backward_kalman=False
    )
    PSIDWrapper._attach_smoother_params(s)
    Y_prep = s.YPrepModel.apply(Y, time_first=True)

    # Refactored smoother
    _, _, Xs_ref = PSIDWrapper._rts_smooth(s, Y_prep)

    # Manual reference using the documented formula
    allXp, _, allXf = s.kalman(Y_prep)[0:3]
    Cs = np.asarray(s.Cs)
    N = allXf.shape[0]
    Xs_manual = np.zeros_like(allXf)
    Xs_manual[-1, :] = allXf[-1, :]
    for t in range(N - 2, -1, -1):
        Xs_manual[t, :] = allXf[t, :] + Cs @ (Xs_manual[t + 1, :] - allXp[t + 1, :])

    diff = np.max(np.abs(Xs_ref - Xs_manual))
    assert diff < 1e-12, f"RTS smoother mismatch: max|diff|={diff:g}"
    print(f"  [ok] RTS smoother matches manual reference (max|diff|={diff:g})")


if __name__ == "__main__":
    tests = [
        test_parity_upstream_with_mods_disabled,
        test_dare_fix_produces_finite_pp,
        test_eigenvalue_damping,
        test_smoother_cz_refit,
        test_smoother_params_attached,
        test_rts_smoother_matches_ref,
    ]
    for t in tests:
        print(f"\n>> {t.__name__}")
        t()
    print("\nAll tests passed.")
