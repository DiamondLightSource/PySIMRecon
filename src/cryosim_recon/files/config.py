from __future__ import annotations
import logging
from pathlib import Path
from configparser import RawConfigParser
from typing import TYPE_CHECKING

from ..settings import WavelengthSettings


if TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Generator

logger = logging.getLogger(__name__)

__PARSER_KWARGS = {"inline_comment_prefixes": "#;"}


def read_input_config(input_config: str | PathLike[str]) -> RawConfigParser:
    config_parser = RawConfigParser(**__PARSER_KWARGS)
    config_parser.read(input_config)
    return config_parser


def get_wavelength_settings(
    otfs_config_path: str | PathLike[str],
) -> Generator[WavelengthSettings, None, None]:

    def get_from_section(
        section_name: str, config_parser: RawConfigParser
    ) -> Generator[tuple[int, Path], None, None]:
        for wavelength, path in config_parser.items(section_name):
            try:
                wavelength = wavelength.strip().lower()
                if wavelength.endswith("nm"):
                    logger.debug("Trimming 'nm' from %s", wavelength)
                    # Handle nm endings but not other units
                    wavelength = wavelength[:-2].strip()
                wavelength = int(wavelength)
            except Exception:
                logger.warning(
                    "'%s' is not a valid wavelength (must be an integer)", wavelength
                )
                continue
            try:
                path = Path(path.strip())
                if not path.is_file():
                    raise FileNotFoundError(
                        f"No {section_name.upper()} file found for {wavelength} at {path}"
                    )
                yield wavelength, path
            except Exception as e:
                logging.warning(
                    "%i %s file path error: %s", wavelength, section_name.upper(), e
                )
                continue

    config_parser = RawConfigParser()
    config_parser.read_file(otfs_config_path)
    wavelengths = set()
    for wavelength, otf_path in get_from_section("otfs", config_parser):
        wavelengths.add(wavelength)
        yield WavelengthSettings(wavelength, otf_path, None)
    for wavelength, psf_path in get_from_section("psfs", config_parser):
        if wavelength in wavelengths:
            # OTFs override PSFs
            logging.info("Ignoring PSF as OTF was given for %i", wavelength)
            continue
        yield WavelengthSettings(wavelength, None, psf_path)


def get_wavelength_config(
    defaults_config_path: str | PathLike[str],
    wavelengths_config_path: str | PathLike[str],
    wavelength: str,
    output_path: str | PathLike[str],
) -> Path | None:
    config_parser = RawConfigParser(**__PARSER_KWARGS)
    # Get defaults as baseline
    config_parser = _read_defaults_config(
        config_parser, defaults_config_path, wavelength
    )
    # Overwrite & add any options from the wavelength config
    config_parser = _read_wavelength_config(config_parser, wavelengths_config_path)
    # Clear sections that aren't for this wavelength:
    for section in config_parser.sections:
        if section != str(wavelength):
            config_parser.remove_section(section)
    text = "\n".join(["=".join(item) for item in config_parser.items(str(wavelength))])
    with open(output_path, "w") as f:
        f.write(text)
    return Path(output_path)


def _read_defaults_config(
    config_parser: RawConfigParser,
    defaults_config_path: str | PathLike[str],
    wavelength: int,
) -> RawConfigParser:
    # sirecon configs don't have section titles
    with open(defaults_config_path, "r") as f:
        # Ensure this is read to the correct wavelength section
        config_contents = f"[{wavelength}]\n" + f.read()
    config_parser.read_string(config_contents)
    return config_parser


def _read_wavelength_config(
    config_parser: RawConfigParser, wavelengths_config_path: str | PathLike[str]
) -> RawConfigParser:
    config_parser.read_file(wavelengths_config_path)
    return config_parser
