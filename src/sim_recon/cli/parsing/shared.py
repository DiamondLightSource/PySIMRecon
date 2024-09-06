from __future__ import annotations
import argparse
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ...settings.formatting import SettingFormat


def add_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )


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


def add_override_args_from_formatters(
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
