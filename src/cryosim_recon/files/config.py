from __future__ import annotations
import logging
from pathlib import Path
from os.path import abspath
from configparser import RawConfigParser
from typing import TYPE_CHECKING, Any

from ..settings import WavelengthSettings
from ._setting_formatting import FORMATTERS

if TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Generator

logger = logging.getLogger(__name__)


__OTF_LOCATIONS_SECTION = "otfs"
__PSF_LOCATIONS_SECTION = "psfs"
__PARSER_KWARGS = {"inline_comment_prefixes": "#;"}
__DIRECTORY_KEY = "directory"
__RECON_CONFIG_SECTION = "recon config"
__OTF_CONFIG_SECTION = "otf config"


def read_input_config(input_config: str | PathLike[str]) -> RawConfigParser:
    config_parser = RawConfigParser(**__PARSER_KWARGS)
    config_parser.read(input_config)
    return config_parser


def get_wavelength_settings(
    defaults_config_path: str | PathLike[str],
    wavelengths_config_path: str | PathLike[str],
) -> Generator[WavelengthSettings, None, None]:
    def parse_wavelength_key(key: str) -> int:
        key = key.strip().lower()
        if key.endswith("nm"):
            logger.debug("Trimming 'nm' from %s", key)
            # Handle nm endings but not other units
            key = key[:-2].strip()
        return int(key)

    def get_from_section(
        section_name: str, config_parser: RawConfigParser
    ) -> Generator[tuple[int, Path], None, None]:
        directory = config_parser.get(section_name, __DIRECTORY_KEY, fallback=None)
        for key, path in config_parser.items(section_name):
            if key == __DIRECTORY_KEY:
                # Ignore __DIRECTORY_KEY
                continue
            try:
                wavelength = parse_wavelength_key(key)
            except Exception:
                logger.warning(
                    "'%s' is not a valid wavelength (must be an integer)", wavelength
                )
                continue
            try:
                if directory is not None:
                    path = Path(directory) / path.strip()
                else:
                    # If directory is not specified, require absolute path
                    path = Path(path.strip())
                if not path.is_file():
                    raise FileNotFoundError(
                        f"No {section_name.upper()} file found for {wavelength} at {path}"
                    )
                # Ensure returned path is absolute
                yield wavelength, path.absolute()
            except Exception as e:
                logging.warning(
                    "%i %s file path error: %s", wavelength, section_name.upper(), e
                )
                continue

    config_parser = RawConfigParser(**__PARSER_KWARGS)
    config_parser.read(defaults_config_path)
    # Get defaults as baseline
    defaults_dict = get_default_recon_kwargs(defaults_config_path)
    # Get otf parameters:
    otf_config = get_otf_kwargs(config_parser)

    config_parser = RawConfigParser()
    config_parser.read_file(wavelengths_config_path)
    wavelengths = set()
    for wavelength, otf_path in get_from_section(
        __OTF_LOCATIONS_SECTION, config_parser
    ):
        wavelengths.add(wavelength)

        config = defaults_dict.copy()
        if config_parser.has_section(str(wavelength)):
            config.update(_config_section_to_dict(config_parser, str(wavelength)))

        yield WavelengthSettings(wavelength, otf=otf_path, config=config)
    for wavelength, psf_path in get_from_section(
        __PSF_LOCATIONS_SECTION, config_parser
    ):
        if wavelength in wavelengths:
            # OTFs override PSFs
            logging.info("Ignoring PSF as OTF was given for %i", wavelength)
            continue

        config = defaults_dict.copy()
        if config_parser.has_section(str(wavelength)):
            config.update(_config_section_to_dict(config_parser, str(wavelength)))

        yield WavelengthSettings(
            wavelength, psf=psf_path, config=config, otf_config=otf_config
        )


def create_wavelength_config(
    output_path: str | PathLike[str],
    otf_path: str | PathLike[str],
    **config_kwargs: Any,
) -> Path | None:
    # Add otf_file that is expected by ReconParams
    config_kwargs["otf_file"] = str(abspath(otf_path))
    # Convert to string so it can be written without a section, like sirecon expects
    text = ""
    for key, value in config_kwargs.items():
        if isinstance(value, (tuple, list)):
            # Comma separated values
            value = ",".join(str(i) for i in value)
        text += f"{key}={value}\n"

    with open(output_path, "w") as f:
        f.write(text)
    return Path(output_path)


def get_default_recon_kwargs(
    config_parser: RawConfigParser,
) -> dict[str, str]:
    return _config_section_to_dict(config_parser, section_name=__RECON_CONFIG_SECTION)


def get_otf_kwargs(
    config_parser: RawConfigParser,
) -> dict[str, str]:
    return _config_section_to_dict(config_parser, section_name=__OTF_CONFIG_SECTION)


def _config_section_to_dict(config_parser: RawConfigParser, section_name: str):
    kwargs = {}
    for key, value in config_parser.items(section_name):
        if value is None or key not in FORMATTERS:
            logger.debug("Option %s=%s is invalid and will be ignored", key, value)
            continue
        setting_format = FORMATTERS.get(value)
        if setting_format.split:
            kwargs[key] = tuple(
                setting_format.conv(s.strip()) for s in value.split(",") if s.strip()
            )
        else:
            kwargs[key] = setting_format.conv(value.strip())
    return kwargs
