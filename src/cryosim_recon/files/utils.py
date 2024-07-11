from __future__ import annotations
from pathlib import Path
from uuid import uuid4
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Literal

__OTF_SUFFIX = "_otf"
__RECON_SUFFIX = "_recon"


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
    for i in range(1, max_iter + 1):
        output_path = path.with_suffix(f"_{i}{path.suffix()}")
        if not output_path.exists():
            return output_path
    raise IOError(
        f"Failed to create unique file path after {max_iter} attempts. Final attempt was '{output_path}'."
    )
