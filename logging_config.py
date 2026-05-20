"""Centralized logging with per-module level control and ANSI color output.

Provides unified file + console handlers with:
- Per-module DEBUG/INFO/WARNING level overrides (DEFAULT_LEVELS dict)
- Automatic console DEBUG activation if any module requests DEBUG
- ANSI color formatting for console (disable via LOG_COLOR=0)
- File logging to results_dir/simulation.log

Environment variables:
  LOG_LEVEL           - Base console level intent (DEBUG|INFO|WARNING|ERROR)
  LOG_CONSOLE_LEVEL   - Force console level (overrides auto-detection)
  LOG_COLOR           - Enable (1) or disable (0) ANSI colors

Typical usage:
    lm = LoggingManager("results/")
    lm.setup()  # Configures root logger and handlers
    logger = logging.getLogger(__name__)  # Per-module logger
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, Optional
import sys

# ---------------------- constants & helpers ----------------------
_ENV_LOG_LEVEL = "LOG_LEVEL"
_ENV_CONSOLE_LEVEL = "LOG_CONSOLE_LEVEL"
_ENV_COLOR = "LOG_COLOR"

_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

# ---------------------- formatter implementations ----------------------
class _ColorFormatter(logging.Formatter):
    """ANSI color formatter for console output.
    
    Applies color codes to level names (DEBUG=grey, INFO=cyan, WARNING=yellow,
    ERROR=red, CRITICAL=magenta) and module names (magenta).
    """
    COLORS = {
        logging.DEBUG: "\x1b[90m",     # grey
        logging.INFO: "\x1b[36m",      # cyan
        logging.WARNING: "\x1b[33m",   # yellow
        logging.ERROR: "\x1b[31m",     # red
        logging.CRITICAL: "\x1b[95m"    # magenta
    }
    MOD = "\x1b[35m"
    RESET = "\x1b[0m"
    def format(self, record):  # type: ignore[override]
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"{self.MOD}{record.name}{self.RESET}"
        return super().format(record)

# ---------------------- core logging manager ----------------------
class LoggingManager:
    """Logging configuration manager with per-module level control.

    Sets up root logger with file + console handlers. Module-specific levels
    defined in DEFAULT_LEVELS dict. Console automatically switches to DEBUG
    if any module requests DEBUG (unless LOG_CONSOLE_LEVEL env var overrides).

    Usage:
        lm = LoggingManager(results_dir="results/")
        lm.setup()
        logger = logging.getLogger("simulation")  # Inherits module-level settings
    """

    DEFAULT_LEVELS: Dict[str, int] = {
        'config': logging.DEBUG,
        'logging_config': logging.DEBUG,
        'simulation': logging.DEBUG,
        'simulation.simulation_core': logging.DEBUG,
        'simulation.simulation_reference': logging.DEBUG,
        'simulation.simulation_dataset': logging.DEBUG,
        'simulation.simulation_loads': logging.DEBUG,
        'simulation.simulation_analysis': logging.DEBUG,
        'simulation.data_mapper': logging.DEBUG,
        'simulation.febio_parser': logging.DEBUG,
        'simulation.storage_manager': logging.DEBUG,
        'simulation.eigensolver': logging.DEBUG,
        'simulation.elasticsolver': logging.DEBUG,
        'simulation.ligamentassembler': logging.DEBUG,
        'material_manager': logging.DEBUG,
        'storage_manager': logging.DEBUG,
        'febio_parser': logging.DEBUG,
        'ligamentassembler': logging.DEBUG,
        'eigensolver': logging.DEBUG,
        'elasticsolver': logging.DEBUG,
        'mode_pairing': logging.DEBUG,
        'mode_analysis': logging.DEBUG,
        'database': logging.DEBUG,
        'utils': logging.DEBUG,
        'anatomy_analyser': logging.DEBUG,
    }

    def __init__(self, results_dir: str, log_levels: Dict[str, int] | None = None):
        self.results_dir = results_dir
        self.log_levels: Dict[str, int] = (log_levels or self.DEFAULT_LEVELS).copy()
        self._configured = False
        self.logger: Optional[logging.Logger] = None
        self._color_support: Optional[bool] = None
        self._log_path: Optional[Path] = None
        self._file_handler: Optional[logging.Handler] = None
        self._console_handler: Optional[logging.Handler] = None

    # -------------------------- public API --------------------------
    @property
    def log_path(self) -> Optional[Path]:
        return self._log_path

    def setup(self) -> logging.Logger:
        if self._configured:
            return self.logger or logging.getLogger(__name__)
        # Prepare paths
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        self._log_path = Path(self.results_dir) / "simulation.log"

        # Reset root handlers (idempotent clean state)
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)

        # Effective levels
        requested_root_level = self._env_level(_ENV_LOG_LEVEL, fallback=logging.INFO)
        console_level = self._determine_console_level(requested_root_level)

        # Root must allow all through so handlers can filter
        root.setLevel(logging.DEBUG)

        # Handlers
        self._file_handler = logging.FileHandler(self._log_path, mode='w')
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(self._build_formatter(colored=False))

        self._console_handler = logging.StreamHandler()
        self._console_handler.setLevel(console_level)
        self._console_handler.setFormatter(self._build_formatter(colored=True))

        root.addHandler(self._file_handler)
        root.addHandler(self._console_handler)

        self._configure_module_loggers()

        self.logger = logging.getLogger("logging_config")
        self.logger.info(
            "Logging initialized: root_intent=%s console_level=%s color=%s log=%s",
            logging.getLevelName(requested_root_level),
            logging.getLevelName(console_level),
            self._color_support,
            self._log_path,
        )
        self._configured = True
        return self.logger

    def add_module(self, module_name: str, level: int | str = logging.INFO) -> None:
        if isinstance(level, str):
            level = _LEVEL_MAP.get(level.upper(), logging.INFO)
        if module_name not in self.log_levels:
            self.log_levels[module_name] = level
        logging.getLogger(module_name).setLevel(level)
        if self.logger:
            self.logger.debug("Added/updated module %s level=%s", module_name, logging.getLevelName(level))

    def set_level(self, module_name: str, level: int | str) -> None:
        if isinstance(level, str):
            level = _LEVEL_MAP.get(level.upper(), logging.INFO)
        self.log_levels[module_name] = level
        logging.getLogger(module_name).setLevel(level)
        if self.logger:
            self.logger.debug("Set level %s for %s", logging.getLevelName(level), module_name)

    def set_global_level(self, level: int | str) -> None:
        if isinstance(level, str):
            level = _LEVEL_MAP.get(level.upper(), logging.INFO)
        for m in list(self.log_levels.keys()):
            self.set_level(m, level)

    def reconfigure_console(self, level: int | str) -> None:
        if isinstance(level, str):
            level = _LEVEL_MAP.get(level.upper(), logging.INFO)
        if self._console_handler:
            self._console_handler.setLevel(level)
        if self.logger:
            self.logger.info("Console log level now %s", logging.getLevelName(level))

    # -------------------------- internals ---------------------------
    def _env_level(self, env_key: str, fallback: int) -> int:
        val = os.getenv(env_key)
        if not val:
            return fallback
        return _LEVEL_MAP.get(val.upper(), fallback)

    def _determine_console_level(self, intent: int) -> int:
        override = os.getenv(_ENV_CONSOLE_LEVEL)
        if override:
            return _LEVEL_MAP.get(override.upper(), logging.DEBUG)
        # auto elevate to DEBUG if any module debug
        if any(level <= logging.DEBUG for level in self.log_levels.values()):
            return logging.DEBUG
        return intent

    def _color_enabled(self) -> bool:
        env = os.getenv(_ENV_COLOR)
        if env is not None:
            return env not in ("0", "false", "False")
        try:
            return sys.stderr.isatty()
        except (AttributeError, OSError):
            return False

    def _build_formatter(self, colored: bool) -> logging.Formatter:
        if self._color_support is None:
            self._color_support = self._color_enabled()
        base_fmt = "%(asctime)s | %(levelname)5s | %(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        if not colored or not self._color_support:
            return logging.Formatter(base_fmt, datefmt=datefmt)
        return _ColorFormatter(base_fmt, datefmt=datefmt)

    def _configure_module_loggers(self) -> None:
        for module_name, level in self.log_levels.items():
            logging.getLogger(module_name).setLevel(level)

# ---------------------- convenience function ----------------------
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
