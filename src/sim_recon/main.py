from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from .files.config import (
    read_config,
    get_defaults_config_path,
    get_recon_kwargs,
    get_otf_kwargs,
    get_channel_configs,
)
from .settings import ConfigManager
from .otfs import convert_psfs_to_otfs
from .recon import run_reconstructions


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike


logger = logging.getLogger(__name__)


def load_configs(config_path: str | PathLike[str]) -> ConfigManager:
    logger.info("Loading configurations from %s...", config_path)

    main_config = read_config(config_path)

    defaults_config_path = get_defaults_config_path(main_config)
    defaults_config = read_config(defaults_config_path)

    default_recon_kwargs = get_recon_kwargs(defaults_config)
    default_otf_kwargs = get_otf_kwargs(defaults_config)

    return ConfigManager(
        defaults_config_path=defaults_config_path,
        default_reconstruction_config=default_recon_kwargs,
        default_otf_config=default_otf_kwargs,
        channel_configs=get_channel_configs(main_config),
    )


def sim_psf_to_otf(
    config_path: str | PathLike[str],
    *psf_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    overwrite: bool = False,
    cleanup: bool = True,
    xy_shape: tuple[int, int] | None = None,
    **otf_kwargs: Any,
) -> None:
    """
    Top level function for converting PSFs to OTFs

    Parameters
    ----------
    config_path : str | PathLike[str]
        Path of the top level config file
    *psf_paths : str | PathLike[str],
        Paths to PSF files (DV expected)
    output_directory : str | PathLike[str] | None, optional
        Directory to save OTFs in (OTFs will be saved with the PSF files if not specified), by default None
    overwrite : bool, optional
        Overwrite files if they already exist, by default False
    cleanup : bool, optional
        Clean up temporary files after OTF conversion, by default True
    xy_shape : tuple[int, int] | None, optional
        Shape to crop PSFs to before conversion (powers of 2 will make for faster processing), by default None
    """
    conf = load_configs(config_path)
    convert_psfs_to_otfs(
        conf,
        *psf_paths,
        output_directory=output_directory,
        overwrite=overwrite,
        cleanup=cleanup,
        xy_shape=xy_shape,
        **otf_kwargs,
    )


def sim_reconstruct(
    config_path: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    overwrite: bool = False,
    cleanup: bool = True,
    stitch_channels: bool = True,
    parallel_process: bool = False,
    **recon_kwargs: Any,
) -> None:
    """
    Top level function for reconstructing SIM data

    Parameters
    ----------
    config_path : str | PathLike[str]
        Path of the top level config file
    *sim_data_paths : str | PathLike[str]
        Paths to SIM data files (DV expected)
    output_directory : str | PathLike[str]
        Directory to save reconstructions in (reconstructions will be saved with the data files if not specified), by default None
    overwrite : bool, optional
        Overwrite files if they already exist, by default False
    cleanup : bool, optional
        Clean up temporary directory and files after reconstruction, by default True
    stitch_channels : bool, optional
        Stitch channels back together after processing (otherwise output will be a separate DV per channel), by default True
    parallel_process : bool, optional
        Run reconstructions in 2 processes concurrently, by default False
    """
    conf = load_configs(config_path)
    logger.info("Starting reconstructions...")
    run_reconstructions(
        conf,
        *sim_data_paths,
        output_directory=output_directory,
        overwrite=overwrite,
        cleanup=cleanup,
        stitch_channels=stitch_channels,
        parallel_process=parallel_process,
        **recon_kwargs,
    )
