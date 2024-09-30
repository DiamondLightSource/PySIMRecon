from __future__ import annotations
import argparse
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ...settings.formatting import SettingFormat


def add_general_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show more logging",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="use_tqdm",
        help="turn off progress bars (only has an effect if tqdm is installed)",
    )


def add_override_args_from_formatters(
    parser: argparse.ArgumentParser, formatters: dict[str, SettingFormat]
) -> None:
    arg_group = parser.add_argument_group(
        "Overrides",
        "Arguments that override configured values. Defaults stated are only used if no value is given or configured.",
    )
    for arg_name, formatter in formatters.items():
        if formatter.nargs == 0:
            arg_group.add_argument(
                f"--{arg_name}",
                action="store_true",
                default=None,
                required=False,
                help=formatter.help_string,
            )
        else:
            arg_group.add_argument(
                f"--{arg_name}",
                type=formatter.conv,
                nargs=formatter.nargs,
                required=False,
                help=formatter.help_string,
            )
