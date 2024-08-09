from __future__ import annotations
import argparse
from typing import TYPE_CHECKING

from ..settings.formatting import OTF_FORMATTERS, RECON_FORMATTERS

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence
    from ..settings.formatting import SettingFormat


def handle_required(
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
    *required: tuple[str, str],
):
    missing_arguments: list[str] = []
    for print_name, dest in required:
        if getattr(namespace, dest, None) is None:
            missing_arguments.append(print_name)
    if missing_arguments:
        parser.error(
            "the following arguments are required: %s" % ", ".join(missing_arguments)
        )


def _add_override_args_from_formatters(
    parser: argparse.ArgumentParser, formatters: dict[str, SettingFormat]
) -> None:
    arg_group = parser.add_argument_group(
        "Overrides",
        "Arguments that override configured values. Defaults stated are only used if no value is given or configured.",
    )
    for arg_name, formatter in formatters.items():
        if formatter.conv is bool:
            arg_group.add_argument(
                f"--{arg_name}",
                action="store_true",
                default=None,
                required=False,
                help=formatter.description,
            )
        else:
            arg_group.add_argument(
                f"--{arg_name}",
                type=formatter.conv,
                nargs=formatter.count,
                required=False,
                help=formatter.description,
            )


def parse_otf_args(
    args: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(prog="sim-otf", add_help=False)
    parser.add_argument(
        "-c",
        "--config-path",
        dest="config_path",
        help="Path to the root config that specifies the paths to the OTFs and the other configs",
    )
    parser.add_argument(
        "-p",
        "--psf",
        dest="psf_paths",
        nargs="+",
        help="Paths to PSF files to be reconstructed (multiple paths can be given)",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        default=None,
        help="If specified, the output directory that the OTFs will be saved in, otherwise each OTF will be saved in the same directory as its PSF",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If specified, files will be overwritten if they already exist (unique filenames will be used otherwise)",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="If specified, files created during the reconstruction process will not be cleaned up",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If specified, files will be overwritten if they already exist (unique filenames will be used otherwise)",
    )

    namespace, unknown = parser.parse_known_args(args)

    _add_override_args_from_formatters(parser, OTF_FORMATTERS)

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )

    override_namespace = parser.parse_args(unknown)

    handle_required(
        parser,
        namespace,
        ("-c/--config-path", "config_path"),
        ("-p/--psf", "psf_paths"),
    )

    non_override_dests = vars(namespace).keys()
    otf_kwargs = {
        k: v
        for k, v in vars(override_namespace).items()
        if v is not None and k not in non_override_dests
    }

    return namespace, otf_kwargs


def parse_recon_args(
    args: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(prog="sim-recon", add_help=False)
    parser.add_argument(
        "-c",
        "--config-path",
        dest="config_path",
        help="Path to the root config that specifies the paths to the OTFs and the other configs",
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="sim_data_paths",
        nargs="+",
        help="Paths to SIM data files to be reconstructed (multiple paths can be given)",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="The output directory to save reconstructed files in",
    )
    parser.add_argument(
        "--keep-split",
        dest="stitch_channels",
        action="store_false",
        help="If specified, channels will not be stitched back together after reconstruction",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="If specified, files created during the reconstruction process will not be cleaned up",
    )
    parser.add_argument(
        "--parallel",
        dest="parallel_process",
        action="store_true",
        help="If specified, up to 2 processes will be run at a time",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If specified, files will be overwritten if they already exist (unique filenames will be used otherwise)",
    )

    namespace, unknown = parser.parse_known_args(args)

    # Now add override arguments so they show up in --help
    _add_override_args_from_formatters(parser, RECON_FORMATTERS)

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )

    override_namespace = parser.parse_args(unknown)

    handle_required(
        parser,
        namespace,
        ("-c/--config-path", "config_path"),
        ("-d/--data", "sim_data_paths"),
        ("-o/--output-directory", "output_directory"),
    )

    non_override_dests = vars(namespace).keys()
    recon_kwargs = {
        k: v
        for k, v in vars(override_namespace).items()
        if v is not None and k not in non_override_dests
    }

    return namespace, recon_kwargs
