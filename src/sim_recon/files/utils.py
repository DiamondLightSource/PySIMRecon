from __future__ import annotations
import logging
import platform
import os
import sys
import re
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Generator

logger = logging.getLogger(__name__)

OTF_NAME_STUB = "OTF"
RECON_NAME_STUB = "recon"

WINDOWS_FN_SUB = re.compile('[<>:"/\\|?*]')
LINUX_FN_SUB = re.compile("/")
DARWIN_FN_SUB = re.compile("[/:]")


def get_temporary_path(directory: Path, stem: str, suffix: str) -> Path:
    tiff_path = (directory / f"{stem}_{uuid4()}").with_suffix(suffix)
    if not tiff_path.exists():
        return tiff_path
    raise FileExistsError(
        f"Failed to create temporary file as the following already exists: {tiff_path}"
    )


def ensure_unique_filepath(path: Path, max_iter: int = 99) -> Path:
    if not path.exists():
        return path
    if max_iter <= 1:
        raise ValueError("max_iter must be >1")
    for i in range(1, max_iter + 1):
        output_path = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not output_path.exists():
            logger.debug("'%s' was not unique, so '%s' will be used", path, output_path)
            return output_path
    raise IOError(
        f"Failed to create unique file path after {max_iter} attempts. Final attempt was '{output_path}'."  # type: ignore[reportPossiblyUnboundVariable]
    )


def ensure_valid_filename(filename: str) -> str:
    rstrip = " "
    system = platform.system()
    if system == "Windows":
        invalid_chars = WINDOWS_FN_SUB
        rstrip = " ."
    elif system == "Linux":
        invalid_chars = LINUX_FN_SUB
    elif system == "Darwin":
        invalid_chars = DARWIN_FN_SUB
    else:
        raise OSError(f"{system} is not a supported system")

    new_filename = filename.rstrip(rstrip)
    new_filename = re.sub(invalid_chars, "_", new_filename)

    if filename != new_filename:
        logger.debug(
            "Removed invalid filename characters: '%s' is now '%s'",
            filename,
            new_filename,
        )

    return new_filename


@contextmanager
def redirect_output_to(file_path: str | PathLike[str]) -> Generator[None, None, None]:
    # Can't use contextlib's redirect_stdout and redirect_stderr as the C++ output isn't captured
    file_path = Path(file_path)
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        f = file_path.open("w+")
        f_fd = f.fileno()
        os.dup2(f_fd, stdout_fd)
        os.dup2(f_fd, stderr_fd)
        yield
    finally:
        # Reset stdout and stderr file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
