from __future__ import annotations
import logging
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import (
        TypeVar,
        Any,
        Callable,
        Protocol,
    )
    from collections.abc import Iterable, Generator
    from contextlib import _GeneratorContextManager

    _T = TypeVar("_T")

    class _ProgressWrapperProtocol(Protocol):

        def __call__(
            self, iterable: Iterable[_T], *args: Any, **kwargs: Any
        ) -> Iterable[_T]: ...


__all__ = ("set_use_tqdm", "get_progress_wrapper", "get_logging_redirect")

logger = logging.getLogger(__name__)


def _passthrough_wrapper(
    iterable: Iterable[_T], *args: Any, **kwargs: Any
) -> Iterable[_T]:
    return iterable


@contextmanager
def _passthrough_logging_redirect() -> Generator[None, None, None]:
    try:
        yield None
    finally:
        pass


class ProgressManager:
    _use_tqdm = False
    progress_wrapper: _ProgressWrapperProtocol = _passthrough_wrapper
    logging_redirect = _passthrough_logging_redirect

    @classmethod
    def set_use_tqdm(cls, v: bool) -> None:
        cls._use_tqdm = v
        cls.update_progress_wrapper()

    @classmethod
    def get_use_tqdm(cls) -> bool:
        return cls._use_tqdm

    @classmethod
    def update_progress_wrapper(cls):
        if cls.get_use_tqdm():
            try:
                from tqdm import tqdm
                from tqdm.contrib.logging import logging_redirect_tqdm

                def tqdm_progress_wrapper(
                    iterable: Iterable[_T], *args: Any, **kwargs: Any
                ) -> Iterable[_T]:
                    return tqdm(  # type: ignore[call-overload]
                        iterable,
                        *args,
                        **kwargs,
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )

                cls.progress_wrapper = tqdm_progress_wrapper
                cls.logging_redirect = logging_redirect_tqdm
                return

            except ImportError:
                logger.warning("tqdm not available, cannot monitor progress")

        cls.progress_wrapper = _passthrough_wrapper
        cls.logging_redirect = _passthrough_logging_redirect


def set_use_tqdm(v: bool) -> None:
    ProgressManager.set_use_tqdm(v)


def get_progress_wrapper() -> _ProgressWrapperProtocol:
    return ProgressManager.progress_wrapper


def get_logging_redirect() -> Callable[[], _GeneratorContextManager[None]]:
    return ProgressManager.logging_redirect
