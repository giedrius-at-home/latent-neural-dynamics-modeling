"""Per-trial RMSE (z-scored) as a pandas DataFrame for Streamlit / HTML reports."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RmseRow:
    condition: str
    rmse_psid: float
    rmse_dpad: float
    rmse_varma: float


def rmse_callout_dataframe(row_off: RmseRow, row_on: RmseRow) -> pd.DataFrame:
    """Two rows (DBS-OFF, DBS-ON) × model columns; suitable for `st.dataframe`."""
    df = pd.DataFrame(
        {
            "Condition": [row_off.condition, row_on.condition],
            "PSID": [row_off.rmse_psid, row_on.rmse_psid],
            "DPAD": [row_off.rmse_dpad, row_on.rmse_dpad],
            "VARMA": [row_off.rmse_varma, row_on.rmse_varma],
        }
    )
    return df.round(3)
