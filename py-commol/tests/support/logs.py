"""Send the messages of commol to the standard error stream during a test run."""

import logging
from typing import TextIO


def add_stderr_logger(level: int = logging.INFO) -> logging.StreamHandler[TextIO]:
    """
    Add a standard error handler to the logger ``commol``.

    The logger takes ``level`` and stops propagating, so its messages show
    once, with their time and level.

    Parameters
    ----------
    level : int
        The level of the logger ``commol``.

    Returns
    -------
    logging.StreamHandler[TextIO]
        The handler it adds.
    """
    logger = logging.getLogger("commol")
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return handler
