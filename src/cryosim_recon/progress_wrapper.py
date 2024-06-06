from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar("T")


logger = logging.getLogger(__name__)

try:
    import tqdm

    progress_wrapper = tqdm
except ImportError:
    logger.warning("tqdm not available, cannot monitor progress")

    def progress_wrapper(x: T, *args, **kwargs) -> T:
        return x
