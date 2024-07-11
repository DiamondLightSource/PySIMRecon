from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from .files.config import (
    read_config,
    get_defaults_config_path,
    get_recon_kwargs,
    get_otf_kwargs,
    get_wavelength_settings,
)
from .settings import SettingsManager
from .otfs import convert_psfs_to_otfs
from .recon import run_reconstructions


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike


logger = logging.getLogger(__name__)


def load_settings(config_path: str | PathLike[str]) -> SettingsManager:
    main_config = read_config(config_path)

    defaults_config_path = get_defaults_config_path(main_config)
    defaults_config = read_config(defaults_config_path)

    default_recon_kwargs = get_recon_kwargs(defaults_config)
    default_otf_kwargs = get_otf_kwargs(defaults_config)

    return SettingsManager(
        defaults_config_path=defaults_config_path,
        default_reconstruction_config=default_recon_kwargs,
        default_otf_config=default_otf_kwargs,
        wavelength_settings=get_wavelength_settings(main_config),
    )


def sim_psf_to_otf(
    config_path: str | PathLike[str],
    *psf_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    ensure_unique_path: bool = True,
    **otf_kwargs: Any,
) -> None:
    if otf_kwargs is None:
        otf_kwargs = {}

    settings = load_settings(config_path)
    convert_psfs_to_otfs(
        settings,
        *psf_paths,
        output_directory=output_directory,
        ensure_unique_path=ensure_unique_path,
        **otf_kwargs,
    )


def sim_reconstruct(
    config_path: str | PathLike[str],
    output_directory: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
    stitch_channels: bool = True,
    cleanup: bool = False,
    **recon_kwargs: Any,
) -> None:
    if recon_kwargs is None:
        recon_kwargs = {}

    settings = load_settings(config_path)
    logger.info("Starting reconstructions")
    run_reconstructions(
        output_directory,
        *sim_data_paths,
        settings=settings,
        stitch_channels=stitch_channels,
        cleanup=cleanup,
        **recon_kwargs,
    )
