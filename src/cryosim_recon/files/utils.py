from __future__ import annotations

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
