from __future__ import annotations
import logging
import os
import inspect
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

from pycudasirecon import make_otf  # type: ignore[import-untyped]

from .files.dv import read_dv, dv_to_temporary_tiff
from .files.utils import create_filename, ensure_unique_filepath
from .settings import SettingsManager
from .progress import progress_wrapper

if TYPE_CHECKING:
    from typing import Any, Literal
    from os import PathLike

logger = logging.getLogger(__name__)


def _get_single_channel_wavelength(psf_path: str | PathLike[str]) -> int:
    with read_dv(psf_path) as f:
        waves = (
            f.hdr.wave1,
            f.hdr.wave2,
            f.hdr.wave3,
            f.hdr.wave4,
            f.hdr.wave5,
        )
    waves = (w for w in waves if w)  # Trim 0s
    assert (
        len(set(waves)) == 1
    ), f"PSFs must be single channel but {psf_path} has wavelengths: {', '.join(str(w) for w in waves)}"
    return int(waves[0])


def convert_psfs_to_otfs(
    settings: SettingsManager,
    *psf_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    ensure_unique_path: bool = True,
    **kwargs,
) -> list[Path]:
    completed_otfs: list[Path] = []
    failed_psfs: list[Path] = []
    logger.info("Checking for PSFs to be converted to OTFs...")
    for psf_path in progress_wrapper(psf_paths, desc="PSF to OTF conversions"):
        try:
            wavelength = _get_single_channel_wavelength(psf_path)
            otf_path = psf_path_to_otf_path(
                psf_path=psf_path,
                output_directory=output_directory,
                ensure_unique=ensure_unique_path,
            )
            if otf_path.is_file():
                logger.info(
                    "Skipping PSF to OTF conversion: "
                    "OTF file %s already exists and PSF file %s has not been modified since it was created",
                    otf_path,
                    psf_path,
                )
                continue
            otf_kwargs = settings.get_otf_config(wavelength)
            otf_kwargs.update(kwargs)
            otf_path = psf_to_otf(
                psf_path=psf_path,
                otf_path=otf_path,
                wavelength=wavelength,
                overwrite=True,
                **otf_kwargs,
            )
        except Exception:
            logger.error("Error during PSF to OTF conversion for '%s'", exc_info=True)
        if otf_path is None:
            failed_psfs.append(psf_path)
        else:
            completed_otfs.append(otf_path)

    if failed_psfs:
        logger.warning(
            "OTF creation failed for the following PSFs:\n%s",
            "\n".join(str(fp) for fp in failed_psfs),
        )
    if completed_otfs:
        logger.info("OTFs created:\n%s", "\n".join(str(fp) for fp in completed_otfs))
    else:
        logger.warning("No OTFs were successfully created")
    return completed_otfs


def psf_to_otf(
    psf_path: str | PathLike[str],
    otf_path: str | PathLike[str],
    overwrite: bool = False,
    **kwargs: Any,
) -> Path | None:

    logger.info("Making OTF file %s from PSF file %s", otf_path, psf_path)
    otf_exists = os.path.isfile(otf_path)
    if otf_exists:
        if overwrite:
            logger.warning("Overwriting file %s", otf_path)
            os.unlink(otf_path)
        else:
            raise FileExistsError(f"File {otf_path} already exists")

    make_otf_kwargs: dict[str, Any] = dict(
        inspect.signature(make_otf).parameters.items()
    )

    for k, v in kwargs.items():
        # Only use kwargs that are accepted by make_otf
        if k in make_otf_kwargs:
            make_otf_kwargs[k] = v

    with dv_to_temporary_tiff(psf_path) as tiff_path:
        make_otf_kwargs["psf"] = tiff_path
        make_otf_kwargs["out_file"] = str(otf_path)
        make_otf(**make_otf_kwargs)

    if not os.path.isfile(otf_path):
        logger.error("Failed to create OTF file %s", otf_path)
        return None
    logger.info("Created OTF '%s'", otf_path)
    return Path(otf_path)


def psf_path_to_otf_path(
    psf_path: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    suffix: Literal[".tiff"] = ".tiff",
    ensure_unique: bool = False,
    max_path_iter: int = 99,
) -> Path:
    psf_path = Path(psf_path)
    timestamp = datetime.fromtimestamp(psf_path.stat().st_mtime).isoformat(
        timespec="microseconds"
    )

    if output_directory is None:
        output_directory = psf_path.parent
    else:
        output_directory = Path(output_directory)

    file_stem = f"{create_filename(stem=psf_path.stem, file_type='OTF')}_{timestamp}"
    output_path = output_directory / f"{file_stem}{suffix}"

    if ensure_unique:
        return ensure_unique_filepath(output_path, max_iter=max_path_iter)
    return output_path
