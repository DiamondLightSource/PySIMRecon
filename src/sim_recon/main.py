from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from .files.config import (
    read_config,
    get_defaults_config_path,
    get_otf_and_recon_kwargs,
    get_channel_configs,
)
from .settings import ConfigManager
from .otfs import convert_psfs_to_otfs
from .recon import run_reconstructions, run_single_reconstruction


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from pathlib import Path
    from multiprocessing.pool import Pool
    from .recon import OutputFileTypes


logger = logging.getLogger(__name__)


def load_configs(
    config_path: str | PathLike[str] | None,
    otf_overrides: dict[int, Path] | None = None,
) -> ConfigManager:
    logger.info("Loading configurations from %s...", config_path)

    if config_path is None:
        return ConfigManager(
            None, channel_configs=get_channel_configs(None, otf_overrides)
        )

    main_config = read_config(config_path)

    defaults_config_path = get_defaults_config_path(main_config)
    default_otf_kwargs, default_recon_kwargs = get_otf_and_recon_kwargs(
        defaults_config_path
    )

    return ConfigManager(
        defaults_config_path=defaults_config_path,
        default_reconstruction_config=default_recon_kwargs,
        default_otf_config=default_otf_kwargs,
        channel_configs=get_channel_configs(main_config, otf_overrides),
    )


def sim_psf_to_otf(
    *psf_paths: str | PathLike[str],
    config_path: str | PathLike[str] | None = None,
    output_directory: str | PathLike[str] | None = None,
    overwrite: bool = False,
    cleanup: bool = True,
    xy_shape: tuple[int, int] | None = None,
    xy_centre: tuple[float, float] | None = None,
    **otf_kwargs: Any,
) -> None:
    """
    Top level function for converting PSFs to OTFs

    Parameters
    ----------
    *psf_paths : str | PathLike[str],
        Paths to PSF files (DV expected)
    config_path : str | PathLike[str] | None, optional
        Path of the top level config file, by default None
    output_directory : str | PathLike[str] | None, optional
        Directory to save OTFs in (OTFs will be saved with the PSF files if not specified), by default None
    overwrite : bool, optional
        Overwrite files if they already exist, by default False
    cleanup : bool, optional
        Clean up temporary files after OTF conversion, by default True
    xy_shape : tuple[int, int] | None, optional
        Shape to crop PSFs to before conversion (powers of 2 will make for faster processing), by default None
    xy_centre : tuple[float, float] | None, optional
        The 0-indexed coordinates the image will be cropped around if xy_shape is given (the image centre is used if not specified), by default None
    """
    conf = load_configs(config_path)
    convert_psfs_to_otfs(
        conf,
        *psf_paths,
        output_directory=output_directory,
        overwrite=overwrite,
        cleanup=cleanup,
        xy_shape=xy_shape,
        xy_centre=xy_centre,
        **otf_kwargs,
    )


def sim_reconstruct(
    *sim_data_paths: str | PathLike[str],
    config_path: str | PathLike[str] | None = None,
    output_directory: str | PathLike[str] | None = None,
    processing_directory: str | PathLike[str] | None = None,
    otf_overrides: dict[int, Path] | None = None,
    overwrite: bool = False,
    cleanup: bool = True,
    stitch_channels: bool = True,
    allow_missing_channels: bool = False,
    output_file_type: OutputFileTypes = "dv",
    parallel_process: bool = False,
    **recon_kwargs: Any,
) -> None:
    """
    Top level function for reconstructing SIM data

    Parameters
    ----------
    *sim_data_paths : str | PathLike[str]
        Paths to SIM data files (DV expected)
    config_path : str | PathLike[str] | None, optional
        Path of the top level config file, by default None
    output_directory : str | PathLike[str] | None, optional
        Directory to save reconstructions in (reconstructions will be saved with the data files if not specified), by default None
    processing_directory : str | PathLike[str] | None, optional
        The directory in which subdirectories of temporary files will be stored for processing (otherwise the output directory will be used), by default None
    otf_overrides : dict[int, Path] | None, optional
        A dictionary with emission wavelengths in nm as keys and paths to OTF files as values (these override configured OTFs), by default None
    overwrite : bool, optional
        Overwrite files if they already exist, by default False
    cleanup : bool, optional
        Clean up temporary directory and files after reconstruction, by default True
    stitch_channels : bool, optional
        Stitch channels back together after processing (otherwise output will be a separate DV per channel), by default True
    parallel_process : bool, optional
        Run reconstructions in 2 processes concurrently, by default False
    allow_missing_channels: bool, optional
        Attempt reconstruction of other channels in a multi-channel file if one or more are not configured, by default False
    output_file_type: Literal["dv", "tiff"], optional
        File type that output images will be saved as, by default "dv"
    """
    conf = load_configs(config_path, otf_overrides=otf_overrides)
    logger.info("Starting reconstructions...")
    run_reconstructions(
        conf,
        *sim_data_paths,
        output_directory=output_directory,
        processing_directory=processing_directory,
        overwrite=overwrite,
        cleanup=cleanup,
        stitch_channels=stitch_channels,
        allow_partial=allow_missing_channels,
        output_file_type=output_file_type,
        parallel_process=parallel_process,
        **recon_kwargs,
    )


def sim_reconstruct_single(
    sim_data_path: str | PathLike[str],
    config_path: str | PathLike[str] | None = None,
    output_directory: str | PathLike[str] | None = None,
    processing_directory: str | PathLike[str] | None = None,
    otf_overrides: dict[int, Path] | None = None,
    overwrite: bool = False,
    cleanup: bool = True,
    stitch_channels: bool = True,
    allow_missing_channels: bool = False,
    output_file_type: OutputFileTypes = "dv",
    multiprocessing_pool: Pool | None = None,
    parallel_process: bool = False,
    **recon_kwargs: Any,
) -> None:
    """
    Top level function for reconstructing SIM data

    Parameters
    ----------
    sim_data_path : str | PathLike[str]
        Path to SIM data files (DV expected)
    config_path : str | PathLike[str] | None, optional
        Path of the top level config file, by default None
    output_directory : str | PathLike[str] | None, optional
        Directory to save reconstructions in (reconstructions will be saved with the data files if not specified), by default None
    processing_directory : str | PathLike[str] | None, optional
        The directory in which the temporary files will be stored for processing (otherwise the output directory will be used), by default None
    otf_overrides : dict[int, Path] | None, optional
        A dictionary with emission wavelengths in nm as keys and paths to OTF files as values (these override configured OTFs), by default None
    overwrite : bool, optional
        Overwrite files if they already exist, by default False
    cleanup : bool, optional
        Clean up temporary directory and files after reconstruction, by default True
    stitch_channels : bool, optional
        Stitch channels back together after processing (otherwise output will be a separate DV per channel), by default True
    allow_missing_channels: bool, optional
        Attempt reconstruction of other channels in a multi-channel file if one or more are not configured, by default False
    output_file_type: Literal["dv", "tiff"], optional
        File type that output images will be saved as, by default "dv"
    multiprocessing_pool : Pool | None, optional
        Multiprocessing pool to run cudasirecon in (`maxtasksperchild=1` is recommended to avoid crashes), by default None
    parallel_process : bool, optional
        Run reconstructions in 2 processes concurrently (ignored if multiprocessing_pool is supplied), by default False
    """
    conf = load_configs(config_path, otf_overrides=otf_overrides)
    logger.info("Starting reconstruction of %s", sim_data_path)
    run_single_reconstruction(
        conf,
        sim_data_path,
        output_directory=output_directory,
        processing_directory=processing_directory,
        overwrite=overwrite,
        cleanup=cleanup,
        stitch_channels=stitch_channels,
        allow_partial=allow_missing_channels,
        output_file_type=output_file_type,
        multiprocessing_pool=multiprocessing_pool,
        parallel_process=parallel_process,
        **recon_kwargs,
    )
