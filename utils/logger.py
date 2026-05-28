"""
utils/logger.py — Loguru-based logger for Procurement Copilot.
Import `logger` from here everywhere — don't use print().
"""

import sys
from loguru import logger
from utils.config import LOG_LEVEL, LOG_DIR

# Remove default handler
logger.remove()

# Pretty console output
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
    colorize=True,
)

# Rotating file log (kept for 7 days)
logger.add(
    LOG_DIR / "procurement_copilot.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
)

__all__ = ["logger"]
