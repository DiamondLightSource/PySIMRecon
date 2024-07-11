from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar, Any
    from collections.abc import Iterable, Generator

    T = TypeVar("T")


logger = logging.getLogger(__name__)


try:
    from tqdm import tqdm

    progress_wrapper = tqdm
except ImportError:
    logger.warning("tqdm not available, cannot monitor progress")

    def progress_wrapper(
        iterable: Iterable[T], *args: Any, **kwargs: Any
    ) -> Iterable[T]:
        return iterable

    @contextmanager
    def logging_redirect() -> Generator[None, None, None]:
        try:
            yield None
        finally:
            pass
