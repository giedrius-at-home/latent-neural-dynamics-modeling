from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import pickle


def extract_psid_matrices(idSys) -> Dict[str, np.ndarray]:
    matrices = {}

    if hasattr(idSys, "A") and idSys.A is not None:
        matrices["A"] = np.array(idSys.A)

    if hasattr(idSys, "Cy") and idSys.Cy is not None:
        matrices["Cy"] = np.array(idSys.Cy)

    if hasattr(idSys, "Cz") and idSys.Cz is not None:
        matrices["Cz"] = np.array(idSys.Cz)

    if hasattr(idSys, "K") and idSys.K is not None:
        matrices["K"] = np.array(idSys.K)

    if hasattr(idSys, "Q") and idSys.Q is not None:
        matrices["Q"] = np.array(idSys.Q)

    if hasattr(idSys, "R") and idSys.R is not None:
        matrices["R"] = np.array(idSys.R)

    if hasattr(idSys, "S") and idSys.S is not None:
        matrices["S"] = np.array(idSys.S)

    return matrices


def extract_dpad_matrices_linear(idSys) -> Dict[str, np.ndarray]:
    matrices = {}

    idSys.restoreModels()

    for attr_name in ["model1_Cy", "model2_Cz"]:
        attr = getattr(idSys, attr_name)
        key = "Cy" if "Cy" in attr_name else "Cz"

        if hasattr(attr, "get_weights"):
            weights = attr.get_weights()
            if len(weights) > 0:
                matrices[key] = weights[0]
        elif hasattr(attr, "layers"):
            for layer in attr.layers:
                if "dense" in layer.name.lower() or "Dense" in type(layer).__name__:
                    weights = layer.get_weights()
                    if len(weights) > 0:
                        matrices[key] = weights[0]
                        break

    return matrices


def extract_dpad_matrices_jacobian(
    idSys, operating_point: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    matrices = {}

    if operating_point is None:
        operating_point = np.zeros((idSys.nx,))

    try:
        import tensorflow as tf

        idSys.restoreModels()

        cells1 = idSys.model1.get_cells()
        cell1 = cells1["fw"] if "fw" in cells1 else cells1

        if hasattr(cell1, "A"):
            x1_tf = tf.Variable(
                operating_point[: idSys.n1].reshape(1, -1), dtype=tf.float32
            )
            with tf.GradientTape() as tape:
                output = cell1.A(x1_tf)
            jacobian = tape.jacobian(output, x1_tf)
            if jacobian is not None:
                matrices["A1_jacobian"] = jacobian.numpy().squeeze()

        cells2 = idSys.model2.get_cells()
        cell2 = cells2["fw"] if "fw" in cells2 else cells2

        if hasattr(cell2, "A"):
            x_tf = tf.Variable(operating_point.reshape(1, -1), dtype=tf.float32)
            with tf.GradientTape() as tape:
                output = cell2.A(x_tf)
            jacobian = tape.jacobian(output, x_tf)
            if jacobian is not None:
                matrices["A2_jacobian"] = jacobian.numpy().squeeze()

        if "A1_jacobian" in matrices and "A2_jacobian" in matrices:
            A_jac = np.zeros((idSys.nx, idSys.nx))
            A_jac[: idSys.n1, : idSys.n1] = matrices["A1_jacobian"]
            A_jac[idSys.n1 :, :] = matrices["A2_jacobian"]
            matrices["A_jacobian"] = A_jac

    except Exception as e:
        print(f"Warning: Could not compute Jacobian matrices: {e}")

    return matrices


def load_model_matrices(
    model_path: Path, model_type: str = "auto"
) -> Tuple[Dict[str, np.ndarray], str]:
    with open(model_path, "rb") as f:
        idSys = pickle.load(f)

    if model_type == "auto":
        if hasattr(idSys, "model1") or hasattr(idSys, "model2"):
            model_type = "dpad"
        else:
            model_type = "psid"

    if model_type == "psid":
        matrices = extract_psid_matrices(idSys)
    elif model_type == "dpad":
        matrices = extract_dpad_matrices_linear(idSys)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return matrices, model_type


def analyze_eigenvalues(A: np.ndarray, sampling_freq: float = 150.0) -> Dict[str, Any]:
    eigenvalues, eigenvectors = np.linalg.eig(A)

    magnitudes = np.abs(eigenvalues)
    angles = np.angle(eigenvalues)

    frequencies_hz = (angles / (2 * np.pi)) * sampling_freq
    frequencies_hz = np.abs(frequencies_hz)

    decay_rates = -np.log(magnitudes) * sampling_freq

    stable = np.all(magnitudes < 1.0)

    oscillatory_mask = np.abs(np.imag(eigenvalues)) > 1e-6
    oscillatory_freqs = frequencies_hz[oscillatory_mask]
    oscillatory_mags = magnitudes[oscillatory_mask]

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "magnitudes": magnitudes,
        "angles": angles,
        "frequencies_hz": frequencies_hz,
        "decay_rates": decay_rates,
        "stable": stable,
        "oscillatory_mask": oscillatory_mask,
        "oscillatory_freqs": oscillatory_freqs,
        "oscillatory_mags": oscillatory_mags,
        "max_magnitude": np.max(magnitudes),
        "spectral_radius": np.max(magnitudes),
    }


def compare_matrices_across_conditions(
    matrices_on: Dict[str, np.ndarray],
    matrices_off: Dict[str, np.ndarray],
    matrix_name: str = "A",
) -> Dict[str, Any]:
    if matrix_name not in matrices_on or matrix_name not in matrices_off:
        return {}

    A_on = matrices_on[matrix_name]
    A_off = matrices_off[matrix_name]

    diff = A_on - A_off

    frobenius_norm_diff = np.linalg.norm(diff, "fro")
    frobenius_norm_on = np.linalg.norm(A_on, "fro")
    frobenius_norm_off = np.linalg.norm(A_off, "fro")
    relative_diff = frobenius_norm_diff / max(frobenius_norm_on, frobenius_norm_off)

    max_abs_diff = np.max(np.abs(diff))

    return {
        "difference": diff,
        "frobenius_norm_diff": frobenius_norm_diff,
        "frobenius_norm_on": frobenius_norm_on,
        "frobenius_norm_off": frobenius_norm_off,
        "relative_diff": relative_diff,
        "max_abs_diff": max_abs_diff,
    }
