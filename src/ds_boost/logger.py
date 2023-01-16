"""
Practical General Logger Module built upon logging module
Angel Martinez-Tenor 2022
Status: Production ready

Example of high-level usage (caller module / notebook):
>import log
>log = logger.init(subfolder="app_caller", level='INFO')
>log.info(" ---- App Caller ----")

Example of submodule usage:
>log = logger.get_logger(__name__)
>log.warning()

Logging levels: 10 or 'DEBUG', 20 or 'INFO', 30 or 'WARNING', 40 or'ERROR', 50 or'CRITICAL'

Logging Module Reference: https://docs.python.org/3/library/logging.html
"""

from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

APP_LOGGER_NAME = "DS_logger"
LOG_PATH = Path("log")
TIMESTAMP_COMPLETE = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_REDUCED = "%H:%M:%S"
TIMESTAMP_FILENAME = "%Y-%m-%d %H-%M-%S"

global_dict: dict = {"logger": None}  # mutable global variable to store the logger


def init(
    logger_name=APP_LOGGER_NAME,
    filepath=None,
    level=logging.INFO,
    simple_format: bool = True,
    subfolder: Path | str | None = None,
    filename_modifier: str = "",
    save_log=False,
) -> logging.Logger:
    """Initialize the logger for a high-level application.
    Args:
        logger_name (str): Name of the logger. Default: APP_LOGGER_NAME
        filepath (pathlib.Path): Path of the log file. Default: None (default LOG_PATH / filename)
        level (int): Logging level: 10:'DEBUG', 20:'INFO', 30:'WARNING', 40:'ERROR', 50:'CRITICAL'. Default: 20 (INFO)
        simple_format (bool): If True (default) , use a simple format for the log messages (simple timestamp + message)
        subfolder (pathlib.Path|str): Subfolder of the log file (e.g.: test, dev, exp, prod ....). Ignored if filepath
         is given. Default: None
        file_suffix (str): Suffix of the log file (e.g.: release_N). Default: None
        save_log (bool): If True (default), save the log file. If False, only print the log messages in the console.

    Returns:
        logging.Logger: Logger object
    """
    filename_prefix = f"{datetime.now().strftime(TIMESTAMP_FILENAME)}"
    if filename_modifier:
        filename_prefix += f"_{filename_modifier}"
    filename = f"{filename_prefix}.log"

    if not filepath:
        filepath = LOG_PATH / Path(subfolder) / Path(filename) if subfolder else LOG_PATH / Path(filename)

    filepath.parent.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if simple_format:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s \t %(message)s", TIMESTAMP_REDUCED)
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s  \t %(levelname)s - %(message)s", TIMESTAMP_COMPLETE)
    # console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    # file handler
    if filepath and save_log:
        fh = logging.FileHandler(filepath)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    global_dict["logger"] = logger
    sys.excepthook = handle_exception  # type: ignore # log uncaught exceptions
    return logger


def get_logger(module_name) -> logging.Logger:
    """Get the logger descendant of the given module name
    Args:
        module_name (str): Name of the module
    Returns:
        logging.Logger: Logger descendant of the given module name
    """
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)


def get_numeric_logger_from_args(args: str) -> int:
    """Get the numeric logger level from the command line arguments
    Args:
        args: Command line arguments
    Returns:
        int: Numeric logger level
    """
    logging_argparse = ArgumentParser(prog=__file__, add_help=False)
    logging_argparse.add_argument("-l", "--log-level", default="INFO", help="set log level")
    logging_args, _ = logging_argparse.parse_known_args(args)
    loglevel = logging_args.log_level
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    return numeric_level


def handle_exception(exc_type: type, exc_value: NameError, exc_traceback):
    """Log uncaught exceptions (linked to sys.excepthook)
    Args:
        exc_type (type): Type of the exception
        exc_value (NameError): Value of the exception
        exc_traceback: Traceback of the exception
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    if global_dict["logger"] is not None:
        global_dict["logger"].critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
