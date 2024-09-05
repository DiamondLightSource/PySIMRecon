from __future__ import annotations
import logging
import os
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from pycudasirecon import make_otf  # type: ignore[import-untyped]

from .files.images import (
    read_mrc_bound_array,
    get_wavelengths_from_dv,
    dv_to_temporary_tiff,
)
from .files.utils import (
    create_output_path,
    get_temporary_path,
    redirect_output_to,
)
from .settings import ConfigManager
from .progress import get_progress_wrapper, get_logging_redirect

if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from .files.images import Wavelengths

logger = logging.getLogger(__name__)


def _format_makeotf_call(
    psf_path: str | PathLike[str],
    otf_path: str | PathLike[str],
    **kwargs: dict[str, Any],
) -> str:
    settings_list: list[str] = []
    value: Any | list[Any] | tuple[Any, ...]
    for key, value in kwargs.items():
        if key in ("psf", "out_file"):
            # These are passed in separately and are positional only
            continue
        if isinstance(value, bool):
            if value:
                settings_list.append(f"-{key.replace('_', '-')}")
            continue
        elif isinstance(value, (tuple, list)):
            # Comma separated values
            value = " ".join((str(v) for v in value))
        settings_list.append(f"-{key.replace('_', '-')} {str(value)}")
    return f"makeotf \"{psf_path}\" \"{otf_path}\" {' '.join(settings_list)}"


def _get_psf_wavelengths(psf_path: str | PathLike[str]) -> Wavelengths:
    array = read_mrc_bound_array(psf_path)
    wavelengths = tuple(get_wavelengths_from_dv(array.Mrc))
    del array
    assert (
        len(wavelengths) == 1
    ), f"PSFs must be single channel but {psf_path} contains: {'; '.join(str(w) for w in wavelengths)}"
    return wavelengths[0]


def convert_psfs_to_otfs(
    conf: ConfigManager,
    *psf_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    overwrite: bool = False,
    cleanup: bool = True,
    xy_shape: tuple[int, int] | None = None,
    **kwargs: Any,
) -> list[Path]:
    completed_otfs: list[Path] = []
    failed_psfs: list[str | PathLike[str]] = []
    logger.info("Checking for PSFs to be converted to OTFs...")

    logging_redirect = get_logging_redirect()
    progress_wrapper = get_progress_wrapper()

    with logging_redirect():
        for psf_path in progress_wrapper(psf_paths, desc="PSF to OTF conversions"):
            otf_path: Path | None = None
            try:
                wavelengths = _get_psf_wavelengths(psf_path)
                otf_path = create_output_path(
                    psf_path,
                    output_type="otf",
                    suffix=".tiff",
                    output_directory=output_directory,
                    wavelength=wavelengths.emission_nm_int,
                    mod_timestamp=True,
                    ensure_unique=not overwrite,
                )
                otf_kwargs = conf.get_otf_config(wavelengths.emission_nm_int)
                otf_kwargs.update(kwargs)
                otf_path = psf_to_otf(
                    psf_path=psf_path,
                    otf_path=otf_path,
                    wavelength=wavelengths.emission_nm_int,
                    overwrite=overwrite,
                    cleanup=cleanup,
                    xy_shape=xy_shape,
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
    xy_shape: tuple[int, int] | None = None,
    **kwargs: Any,
) -> Path | None:
    otf_path = Path(otf_path)
    psf_path = Path(psf_path)
    logger.info("Generating OTF from %s: %s", otf_path, psf_path)

    make_otf_kwargs: dict[str, Any] = dict(
        inspect.signature(make_otf).parameters.items()
    )

    for k, v in kwargs.items():
        # Only use kwargs that are accepted by make_otf
        if k in make_otf_kwargs:
            make_otf_kwargs[k] = v

    with dv_to_temporary_tiff(
        psf_path,
        get_temporary_path(otf_path.parent, f".{psf_path.stem}", suffix=".tiff"),
        delete=cleanup,
        xy_shape=xy_shape,
        overwrite=overwrite,
    ) as tiff_path:
        make_otf_kwargs["psf"] = str(tiff_path)
        make_otf_kwargs["out_file"] = str(otf_path)

        with redirect_output_to(otf_path.with_suffix(".log")):
            print(
                "%s\n%s"
                % (
                    _format_makeotf_call(tiff_path, otf_path, **make_otf_kwargs),
                    "-" * 80,
                )
            )
            make_otf(**make_otf_kwargs)

    if not os.path.isfile(otf_path):
        logger.error(
            "Failed to create OTF file from %s - please check the config",
            psf_path,
        )
        return None
    logger.debug("Created OTF '%s'", otf_path)
    return Path(otf_path)
