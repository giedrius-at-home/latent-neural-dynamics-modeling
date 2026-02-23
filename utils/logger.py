import logging
import sys
from pathlib import Path
from datetime import datetime as dt
from typing import Optional
import warnings


class MarkdownFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        lines = message.split("\n")
        if len(lines) > 2 and all(
            part.startswith("---") or part == ""
            for part in lines[1].strip().split("|")
            if part
        ):
            return "\n".join([lines[0]] + lines[2:])
        return message


class _Tee:

    def __init__(self, original_stream, file_path: Path):
        self._orig = original_stream
        self._file_path = file_path
        self._fh = open(self._file_path, mode="a", buffering=1, encoding="utf-8")

    def write(self, data):
        try:
            self._orig.write(data)
        except Exception:
            pass
        try:
            self._fh.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class Logger:
    def __init__(self, log_dir: str, name: str = "application"):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(name).stem
        self.structured_log_file = self.log_dir / f"{stem}_{ts}.md"
        self.console_log_file = self.log_dir / f"{stem}_{ts}.console.log"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # prevent duplication to root handlers

        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(console_formatter)
        self.logger.addHandler(ch)

        file_formatter = MarkdownFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(self.structured_log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(file_formatter)
        self.logger.addHandler(fh)

        logging.captureWarnings(True)
        warnings.filterwarnings("default")

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._stdout_tee = _Tee(sys.stdout, self.console_log_file)
        self._stderr_tee = _Tee(sys.stderr, self.console_log_file)
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee

        self.logger.info(f"Structured log file: {self.structured_log_file}")
        self.logger.info(f"Console mirror file: {self.console_log_file}")

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def close(self):
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self._stdout_tee.close()
            self._stderr_tee.close()
        except Exception:
            pass


_LOGGER_INSTANCE: Optional[Logger] = None


def setup_logger(log_dir: str, name: str = "application"):
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = Logger(log_dir, name)
    return _LOGGER_INSTANCE


def get_logger():
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = Logger("logs", name="default")
    return _LOGGER_INSTANCE


def logger_executor(log_dir: str, name: str):
    setup_logger(log_dir, name=name)
