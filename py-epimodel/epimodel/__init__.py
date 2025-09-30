import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())


from typing import TextIO  # noqa: E402

from epimodel import constants, context  # noqa: E402

try:
    from epimodel import epimodel_rs
except ImportError as e:
    print(f"Error importing Rust extension: {e}")
    epimodel_rs = None

RUST_AVAILABLE = epimodel_rs is not None

from epimodel.api import ModelBuilder, ModelLoader  # noqa: E402


def add_stderr_logger(level: int = logging.INFO) -> logging.StreamHandler[TextIO]:
    logger = logging.getLogger(__name__)
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return handler


__all__ = [
    "ModelBuilder",
    "ModelLoader",
    "constants", 
    "context",
    "epimodel_rs",
    "RUST_AVAILABLE",
]
