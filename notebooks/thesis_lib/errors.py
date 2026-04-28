"""Lightweight thesis exceptions (keep import graph free of loaders/torch)."""


class ThesisDataError(FileNotFoundError):
    """Missing or incomplete thesis results (strict loading; no silent alternate timestamps)."""
