import os
import logging
from logging import getLogger
from rich.logging import RichHandler

# Disable noisy libraries
os.environ["ABSL_LOGGING_VERBOSITY"] = "3"

# Setup distributed training aware logger
RANK = int(os.environ.get("LOCAL_RANK", "0"))
logger = getLogger("onlineEM")
logger.setLevel(logging.DEBUG if RANK == 0 else logging.WARNING)
logger.propagate = False

# Add Rich handler
logger.addHandler(RichHandler(rich_tracebacks=True, show_time=False, markup=True))

# Silence noisy libraries more aggressively
noisy_loggers = [
    "jax._src.xla_bridge",
    "grain._src",
    "grain",
    "jax._src",
    "jax",
    "absl",
]

for logger_name in noisy_loggers:
    noisy_logger = getLogger(logger_name)
    noisy_logger.setLevel(logging.ERROR)
    noisy_logger.propagate = False

from .trainers import Trainer

__all__ = ["Trainer"]
