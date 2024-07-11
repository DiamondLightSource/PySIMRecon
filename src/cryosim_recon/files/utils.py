from __future__ import annotations
import logging
import platform
import re
from pathlib import Path
from uuid import uuid4
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Literal

logger = logging.getLogger(__name__)

__OTF_SUFFIX = "_otf"
__RECON_SUFFIX = "_recon"

WINDOWS_FN_SUB = re.compile("[<>:\"/\\|?*]")
LINUX_FN_SUB = re.compile("/")
DARWIN_FN_SUB = re.compile("[/:]")


def create_filename(
    stem: str,
    file_type: Literal["OTF", "split", "recon", "config"],
    *,
    wavelength: float | None = None,
    extension: str | None = None,
) -> str:
    match file_type:
        case "OTF":
            suffix = __OTF_SUFFIX
        case "recon":
            suffix = __RECON_SUFFIX
        case "split":
            if wavelength is None:
                raise ValueError("Wavelength must be given for split files")
            suffix = ""
        case "split" | "config":
            suffix = ""
            filename = "{0}_{wavelength}_{}"
        case _:
            raise ValueError("Invalid type %s given", file_type)
    filename = stem
    if wavelength is not None:
        filename += f"_{wavelength}"
    filename += suffix
    if extension is not None:
        # Ensure has . only once
        filename += f".{extension.lstrip(".")}"
    return filename


def get_temporary_path(directory: Path, stem: str, suffix:str) -> Path:
    for _ in range(4):
        tiff_path = (directory / f"{stem}_{uuid4()}").with_suffix(suffix)
        if not tiff_path.exists():
            return tiff_path
    raise FileExistsError(f"Failed to create temporary file with stem '{stem}' and suffix '{suffix}' in '{directory}' due to multiple collisions")

def ensure_unique_filepath(path: Path, max_iter: int=99)-> Path:
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
        f"Failed to create unique file path after {i} attempts. Final attempt was '{output_path}'."  # type: ignore[reportPossiblyUnboundVariable]
    )

def ensure_valid_filename(filename:str) -> str:
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
        logger.debug("Removed invalid filename characters: '%s' is now '%s'")

    return new_filename
