from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING


from .files.config import read_input_config, get_wavelength_settings
from .settings import SettingsManager
from .otfs import convert_psfs_to_otfs
from .recon import run_reconstructions


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike


logger = logging.getLogger(__name__)


def load_settings(config_path: str | PathLike[str]) -> SettingsManager:
    config = read_input_config(config_path)
    defaults_config_path = Path(config.get("configs", "defaults"))
    if not defaults_config_path.is_file():
        raise FileNotFoundError(
            f"Configured defaults config does not exist a {defaults_config_path}"
        )
    wavelengths_config_path = Path(config.get("configs", "wavelengths"))
    if not wavelengths_config_path.is_file():
        raise FileNotFoundError(
            f"Configured wavelengths config does not exist a {wavelengths_config_path}"
        )

    return SettingsManager(
        defaults_config=defaults_config_path,
        wavelengths_config=wavelengths_config_path,
        wavelength_settings=get_wavelength_settings(
            defaults_config_path, wavelengths_config_path
        ),
    )


def run(
    config_path: str | PathLike[str],
    output_directory: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
    stitch_channels: bool = True,
    cleanup: bool = False,
    otf_kwargs: dict[str, Any] | None = None,
    recon_kwargs: dict[str, Any] | None = None,
) -> None:
    if otf_kwargs is None:
        otf_kwargs = {}
    if recon_kwargs is None:
        recon_kwargs = {}

    settings = load_settings(config_path)
    convert_psfs_to_otfs(settings, **otf_kwargs)
    logger.info("Starting reconstructions")
    run_reconstructions(
        output_directory,
        *sim_data_paths,
        settings,
        stitch_channels=stitch_channels,
        cleanup=cleanup,
        **recon_kwargs,
    )


if __name__ == "__main__":
    run(*sys.argv[1:])
