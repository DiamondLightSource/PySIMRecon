from __future__ import annotations
import logging
import os
import inspect
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

from pycudasirecon import make_otf  # type: ignore[import-untyped]

from .files.images import read_dv, dv_to_temporary_tiff
from .files.utils import ensure_unique_filepath, ensure_valid_filename, OTF_NAME_STUB
from .settings import SettingsManager
from .files.config import format_kwargs_as_config
from .progress import get_progress_wrapper, get_logging_redirect

if TYPE_CHECKING:
    from typing import Any, Literal
    from os import PathLike

logger = logging.getLogger(__name__)


def _get_single_channel_wavelength(psf_path: str | PathLike[str]) -> int:
    with read_dv(psf_path) as f:
        waves: tuple[int, ...] = (
            f.hdr.wave1,
            f.hdr.wave2,
            f.hdr.wave3,
            f.hdr.wave4,
            f.hdr.wave5,
        )
    waves = tuple(w for w in waves if w)  # Trim 0s
    assert (
        len(set(waves)) == 1
    ), f"PSFs must be single channel but {psf_path} has wavelengths: {', '.join(str(w) for w in waves)}"
    return int(waves[0])


def convert_psfs_to_otfs(
    settings: SettingsManager,
    *psf_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    overwrite: bool = False,
    cleanup: bool = True,
    **kwargs: Any,
) -> list[Path]:
    completed_otfs: list[Path] = []
    failed_psfs: list[str | PathLike[str]] = []
    logger.info("Checking for PSFs to be converted to OTFs...")

    logging_redirect = get_logging_redirect()
    progress_wrapper = get_progress_wrapper()

    with logging_redirect():
        for psf_path in progress_wrapper(psf_paths, desc="PSF to OTF conversions"):
            psf_path: str | PathLike[str]
            otf_path: Path | None = None
            try:
                wavelength = _get_single_channel_wavelength(psf_path)
                otf_path = psf_path_to_otf_path(
                    psf_path=psf_path,
                    output_directory=output_directory,
                    ensure_unique=not overwrite,
                    wavelength=wavelength,
                )
                otf_kwargs = settings.get_otf_config(wavelength)
                otf_kwargs.update(kwargs)
                otf_path = psf_to_otf(
                    psf_path=psf_path,
                    otf_path=otf_path,
                    wavelength=wavelength,
                    overwrite=overwrite,
                    cleanup=cleanup,
                    **otf_kwargs,
                )
            except Exception:
                logger.error(
                    "Error during PSF to OTF conversion for '%s'",
                    psf_path,
                    exc_info=True,
                )
            if otf_path is None:
                failed_psfs.append(psf_path)
            else:
                completed_otfs.append(otf_path)

        if failed_psfs:
            logger.warning(
                "OTF creation failed for the following PSFs:\n\t%s",
                "\n\t".join(str(fp) for fp in failed_psfs),
            )
        if completed_otfs:
            logger.info(
                "OTFs created:\n\t%s", "\n\t".join(str(fp) for fp in completed_otfs)
            )
        else:
            logger.warning("No OTFs were created")
        return completed_otfs


def psf_to_otf(
    psf_path: str | PathLike[str],
    otf_path: str | PathLike[str],
    overwrite: bool = False,
    cleanup: bool = True,
    **kwargs: Any,
) -> Path | None:
    otf_path = Path(otf_path)
    psf_path = Path(psf_path)
    logger.info("Making OTF file %s from PSF file %s", otf_path, psf_path)
    if otf_path.is_file():
        if overwrite:
            logger.warning("Overwriting file %s", otf_path)
            otf_path.unlink()
        else:
            raise FileExistsError(f"File {otf_path} already exists")

    make_otf_kwargs: dict[str, Any] = dict(
        inspect.signature(make_otf).parameters.items()  # type: ignore[reportUnknownArgumentType]
    )

    for k, v in kwargs.items():
        # Only use kwargs that are accepted by make_otf
        if k in make_otf_kwargs:
            make_otf_kwargs[k] = v

    with dv_to_temporary_tiff(
        psf_path, otf_path.parent, delete=cleanup, xy_shape=(256, 256)
    ) as tiff_path:
        make_otf_kwargs["psf"] = str(tiff_path)
        make_otf_kwargs["out_file"] = str(otf_path)

        logger.info(
            "Calling make_otf with arguments:\n\t%s",
            "\n\t".join(format_kwargs_as_config(make_otf_kwargs)),
        )
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
    wavelength: int | None = None,
    ensure_unique: bool = False,
    max_path_iter: int = 99,
) -> Path:
    psf_path = Path(psf_path)
    # datetime.isoformat fails on Windows due to colons being invalid in paths
    timestamp = datetime.fromtimestamp(psf_path.stat().st_mtime).strftime(
        "%Y%m%d_%H%M%S"
    )

    if output_directory is None:
        output_directory = psf_path.parent
    else:
        output_directory = Path(output_directory)

    file_stem = f"{psf_path.stem}_{wavelength}_{OTF_NAME_STUB}_{timestamp}"
    output_path = output_directory / ensure_valid_filename(f"{file_stem}{suffix}")

    if ensure_unique:
        output_path = ensure_unique_filepath(output_path, max_iter=max_path_iter)
    return output_path
