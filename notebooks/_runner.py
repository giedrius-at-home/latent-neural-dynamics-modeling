"""Run a thesis notebook as a script with fig.show() suppressed.

Usage:
    python notebooks/_runner.py notebooks/thesis_sec2_model_validation.py
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "json"
go.Figure.show = lambda self, *a, **k: None

script = Path(sys.argv[1]).resolve()
code = script.read_text()
exec(
    compile(code, str(script), "exec"),
    {"__name__": "__main__", "__file__": str(script)},
)
