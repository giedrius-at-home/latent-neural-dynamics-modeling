"""PSID / DPAD / VARMA pipelines.

All three frameworks share ``FrameworkPipeline``. Shared helpers live in
``pipelines.utils``. Modal orchestration lives in ``dpad_modal.py``.
"""

from ._base import FrameworkPipeline

__all__ = ["FrameworkPipeline"]
