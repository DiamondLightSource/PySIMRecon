from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Literal


def create_filename(
    stem: str,
    wavelength: float,
    file_type: Literal["OTF", "split", "recon", "config"],
) -> str:
    match file_type:
        case "OTF":
            filename = "{0}_{wavelength}_otf"
        case "split" | "config":
            filename = "{0}_{wavelength}"
        case "recon":
            filename = "{0}_{wavelength}_recon"
        case _:
            raise ValueError("Invalid type %s given", file_type)
    return filename.format(stem, wavelength=wavelength)
