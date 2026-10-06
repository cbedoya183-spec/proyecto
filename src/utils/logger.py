"""
logger.py — Utilidad centralizada de registro para todo el proyecto.
"""

import logging
import sys


def get_logger(name: str = "ml_project") -> logging.Logger:
    """
    Retorna un logger con formato profesional configurado para consola.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
