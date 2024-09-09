from __future__ import annotations
import argparse
from typing import TYPE_CHECKING

from ...settings.formatting import OTF_FORMATTERS
from .shared import (
    add_general_args,
    add_override_args_from_formatters,
    add_help,
    handle_required,
)

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence

__all__ = ("parse_args",)


def parse_args(
    args: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(
        prog="sim-otf",
        description="SIM PSFs to OTFs",
        add_help=False,
    )
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
        help="If specified, files created during the OTF creation process will not be cleaned up",
    )
    parser.add_argument(
        "--shape",
        dest="xy_shape",
        default=None,
        nargs=2,
        type=int,
        help="Takes 2 integers (X Y), specifying the shape to crop PSFs to before converting (powers of 2 are fastest)",
    )

    add_general_args(parser)

    namespace, unknown = parser.parse_known_args(args)

    add_override_args_from_formatters(parser, OTF_FORMATTERS)

    add_help(parser)

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


if __name__ == "__main__":
    parse_args()
