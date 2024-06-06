from __future__ import annotations
import logging
import os
import inspect
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

from pycudasirecon import make_otf  # type: ignore[import-untyped]

from .files.utils import create_filename
from .settings import SettingsManager
from .progress_wrapper import progress_wrapper

if TYPE_CHECKING:
    from typing import Any
    from os import PathLike

# TODO: Use TIFFs are for OTFs as DV files are not supported

logger = logging.getLogger(__name__)


def convert_psfs_to_otfs(settings: SettingsManager, **kwargs) -> None:
    logger.info("Checking for PSFs to be converted to OTFs...")
    for wavelength, wavelength_settings in progress_wrapper(
        settings.wavelengths.items(), desc="PSF to OTF conversions"
    ):
        psf_path = wavelength_settings.psf
        if psf_path is not None:
            otf_path = psf_path_to_otf_path(psf_path=psf_path)
            if otf_path.is_file():
                logging.info(
                    "Skipping PSF to OTF conversion: "
                    "PSF file %s has not been modified since OTF file %s was created",
                    psf_path,
                    otf_path,
                )
                continue
            otf_kwargs = wavelength_settings.otf_config.copy()
            otf_kwargs.update(kwargs)
            psf_to_otf(
                psf_path=psf_path,
                otf_path=otf_path,
                wavelength=wavelength,
                overwrite=True,
                **otf_kwargs,
            )
            wavelength_settings.otf = otf_path


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

    make_otf_kwargs = dict(inspect.signature(make_otf).parameters.items())
    make_otf_kwargs["psf"] = str(psf_path)
    make_otf_kwargs["out_file"] = str(otf_path)

    for k, v in kwargs:
        # Only use kwargs that are accepted by make_otf
        if k in make_otf_kwargs:
            make_otf_kwargs[k] = v

    make_otf(**make_otf_kwargs)


def psf_path_to_otf_path(psf_path: str | PathLike[str]) -> Path:
    psf_path = Path(psf_path)
    timestamp = datetime.fromtimestamp(psf_path.stat().st_mtime).isoformat(
        timespec="microseconds"
    )
    return psf_path.with_stem(
        f"{create_filename(stem=psf_path.stem, file_type='OTF')}_{timestamp}"
    )
