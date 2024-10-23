from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from decimal import Decimal
from configparser import RawConfigParser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Literal
    from collections.abc import Callable

# Using the logic from RawConfigParser for consistency
_boolean_conv = RawConfigParser()._convert_to_boolean  # type: ignore[attr-defined]


def _bool_as_int(v: str) -> int:
    return int(_boolean_conv(v))


def int_range_conversion_wrapper(
    minimum: int | None = None, maximum: int | None = None
) -> Callable[[str], int]:
    def wrappend_conversion_fn(v: str) -> int:
        int_value = int(v)
        if minimum is not None and int_value < minimum:
            raise ValueError(f"Integer greater than or equal to {minimum} is required")
        if maximum is not None and int_value > maximum:
            raise ValueError(f"Integer less than or equal to {maximum} is required")
        return int_value

    return wrappend_conversion_fn


class SettingConverters:
    FLOAT = Decimal
    INT = int
    BOOL = staticmethod(_boolean_conv)
    INT_FROM_BOOL = staticmethod(_bool_as_int)
    INT_POSITIVE = staticmethod(int_range_conversion_wrapper(minimum=0))
    INT_GREATER_THAN_ONE = staticmethod(int_range_conversion_wrapper(minimum=1))
    PATH = Path


class Comparators(Enum):
    GREATER = (">", lambda x, y: x > y)
    GREATER_EQUAL = (">=", lambda x, y: x >= y)
    LESS = ("<", lambda x, y: x < y)
    LESS_EQUAL = ("<=", lambda x, y: x <= y)
    EQUAL = ("==", lambda x, y: x == y)
    NOT_EQUAL = ("!=", lambda x, y: x != y)
    IS = ("is", lambda x, y: x is y)
    IS_NOT = ("is not", lambda x, y: x is not y)


@dataclass(slots=True, frozen=True)
class ConditionalSettingState:
    name: str
    comp: Comparators
    value: Any

    def check_condition_met(self, value: Any) -> bool:
        return self.comp.value[1](value, self.value)

    def __str__(self) -> str:
        string = f"{self.name}"
        if self.comp != Comparators.IS_NOT or self.value is not None:
            string += f" {self.comp.value[0]} {self.value}"
        return string


@dataclass(slots=True, frozen=True)
class SettingFormat:
    conv: Callable[[str], Any]
    nargs: int | Literal["+"] = 1  # "+" matching argparse
    description: str | None = None
    default_value: Any = None
    depends_on: tuple[ConditionalSettingState, ...] | None = None

    @property
    def help_string(self) -> str | None:
        help_string_parts = []
        if self.description is not None:
            help_string_parts.append(self.description)
        if self.default_value is not None:
            help_string_parts.append(f"(default={self.default_value})")
        if self.depends_on is not None:
            help_string_parts.append(
                f"(depends on: {', '.join(str(condition) for condition in self.depends_on)})"
            )
        if help_string_parts:
            return " ".join(help_string_parts)
        return None


OTF_FORMATTERS: dict[str, SettingFormat] = {
    # Note: bools are kept as bools as they are handled by `create_makeotf_command`
    # Shared arguments:
    "nphases": SettingFormat(
        SettingConverters.INT_GREATER_THAN_ONE,
        description="Number of pattern phases per SIM direction",
        default_value=5,
    ),
    "ls": SettingFormat(
        SettingConverters.FLOAT,
        description="Line spacing of SIM pattern in microns",
        default_value=Decimal("0.172"),
    ),
    "na": SettingFormat(
        SettingConverters.FLOAT,
        description="Detection objective's numerical aperture",
        default_value=Decimal("1.2"),
    ),
    "nimm": SettingFormat(
        SettingConverters.FLOAT,
        description="Refractive index of immersion medium",
        default_value=Decimal("1.33"),
    ),
    "background": SettingFormat(
        SettingConverters.FLOAT,
        description="Camera readout background",
        default_value=Decimal(0),
    ),
    # OTF specific args:
    "beaddiam": SettingFormat(
        SettingConverters.FLOAT,
        description="The diameter of the bead in microns",
        default_value=Decimal("0.12"),
    ),
    "angle": SettingFormat(
        SettingConverters.FLOAT,
        description="The k0 vector angle with which the PSF is taken",
        default_value=Decimal(0),
    ),
    "nocompen": SettingFormat(
        SettingConverters.BOOL,
        nargs=0,
        description="Do not perform bead size compensation",
    ),
    "5bands": SettingFormat(
        SettingConverters.BOOL,
        nargs=0,
        description="Output to 5 OTF bands (otherwise higher-order's real and imaginary bands are combined into one output)",
    ),
    "fixorigin": SettingFormat(
        SettingConverters.INT_POSITIVE,
        nargs=2,
        description="The starting and end pixel for interpolation along kr axis",
        default_value=(2, 9),
    ),
    "leavekz": SettingFormat(
        SettingConverters.INT_POSITIVE,
        nargs=3,
        description="Pixels to be retained on kz axis",
        default_value=(0, 0, 0),
    ),
    "I2M": SettingFormat(
        SettingConverters.PATH,
        description="I2M OTF file (input data contains I2M PSF)",
    ),
}

RECON_FORMATTERS: dict[str, SettingFormat] = {
    # Note: bools are stored as integers as this matches the example config
    "ndirs": SettingFormat(
        SettingConverters.INT_GREATER_THAN_ONE,
        description="Number of SIM  directions",
        default_value=3,
    ),
    "nphases": SettingFormat(
        SettingConverters.INT_GREATER_THAN_ONE,
        description="Number of pattern phases per SIM direction",
        default_value=5,
    ),
    "nordersout": SettingFormat(
        SettingConverters.INT_POSITIVE,
        description="Number of output SIM orders (must be <= nphases//2; safe to ignore usually)",
        default_value=0,
    ),
    "angle0": SettingFormat(
        SettingConverters.FLOAT,
        description="Angle of the first SIM angle in radians",
        default_value=Decimal("1.648"),
        depends_on=(ConditionalSettingState("k0angles", Comparators.IS, value=None),),
    ),
    "ls": SettingFormat(
        SettingConverters.FLOAT,
        description="Line spacing of SIM pattern in microns",
        default_value=Decimal("0.172"),
    ),
    "na": SettingFormat(
        SettingConverters.FLOAT,
        description="Detection objective's numerical aperture",
        default_value=Decimal("1.2"),
    ),
    "nimm": SettingFormat(
        SettingConverters.FLOAT,
        description="Refractive index of immersion medium",
        default_value=Decimal("1.33"),
    ),
    "wiener": SettingFormat(
        SettingConverters.FLOAT,
        description="Wiener constant; lower value leads to higher resolution and noise (playing with it extensively is strongly encouraged)",
        default_value=Decimal("0.01"),
    ),
    "otfcutoff": SettingFormat(
        SettingConverters.FLOAT,
        description='OTF threshold below which it\'ll be considered noise and not used in "makeoverlaps"',
        default_value=Decimal("0.006"),
    ),
    "zoomfact": SettingFormat(
        SettingConverters.FLOAT,
        description="Lateral zoom factor in the output over the input images",
        default_value=Decimal(2),
    ),
    "zzoom": SettingFormat(
        SettingConverters.INT_GREATER_THAN_ONE,
        description="Axial zoom factor",
        default_value=1,
    ),
    "background": SettingFormat(
        SettingConverters.FLOAT,
        description="Camera readout background",
        default_value=Decimal(0),
    ),
    "usecorr": SettingFormat(
        SettingConverters.PATH,
        description="Use a flat-field correction file if provided",
    ),
    "forcemodamp": SettingFormat(
        SettingConverters.FLOAT,
        nargs="+",
        description="Force modamps to these values; useful when image quality is low and auto-estimated modamps are below, say, 0.1",
    ),
    "k0angles": SettingFormat(
        SettingConverters.FLOAT,
        nargs=3,
        description="Use these pattern vector k0 angles for all directions (instead of inferring the rest angles from angle0)",
    ),
    "otfRA": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="Use rotationally averaged OTF, otherwise 3/2D OTF is used for 3/2D raw data",
        default_value=0,
    ),
    "otfPerAngle": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="Use one OTF per SIM angle, otherwise one OTF is used for all angles, which is how it's been done traditionally",
        default_value=0,
    ),
    "fastSI": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="SIM image is organized in Z->Angle->Phase order, otherwise Angle->Z->Phase image order is assumed",
        default_value=0,
    ),
    "k0searchAll": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        description="Search for k0 at all time points",
        default_value=0,
    ),
    "norescale": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        description="No bleach correction",
        default_value=0,
    ),
    "equalizez": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="Bleach correction for z",
        default_value=0,
        depends_on=(
            ConditionalSettingState("norescale", Comparators.NOT_EQUAL, value=1),
        ),
    ),
    "equalizet": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="Bleach correction for time",
        default_value=0,
        depends_on=(
            ConditionalSettingState("norescale", Comparators.NOT_EQUAL, value=1),
        ),
    ),
    "dampenOrder0": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="Dampen order-0 in final assembly; do not use for 2D SIM; good choice for high-background images",
        default_value=0,
    ),
    "nosuppress": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        description="Do not suppress DC singularity in the result (good choice for 2D/TIRF data)",
        default_value=0,
    ),
    "nokz0": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="Do not use kz=0 plane of the 0th order in the final assembly (mostly for debug)",
        default_value=0,
    ),
    "gammaApo": SettingFormat(
        SettingConverters.FLOAT,
        description="Output apodization gamma; 1.0 means triangular apo; lower value means less dampening of high-resolution info at the trade-off of higher noise",
        default_value=Decimal(1),
    ),
    "explodefact": SettingFormat(
        SettingConverters.FLOAT,
        description="Artificially explode the reciprocal-space distance between orders by this factor (for debug)",
        default_value=Decimal(1),
    ),
    "nofilterovlps": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        description="Do not filter the overlapping region between bands (for debug)",
        default_value=0,
    ),
    "saveprefiltered": SettingFormat(
        SettingConverters.PATH,
        description="Save separated bands (half Fourier space) into a file and exit (for debug)",
    ),
    "savealignedraw": SettingFormat(
        SettingConverters.PATH,
        description="Save drift-fixed raw data (half Fourier space) into a file and exit (for debug)",
    ),
    "saveoverlaps": SettingFormat(
        SettingConverters.PATH,
        description="Save overlap0 and overlap1 (real-space complex data) into a file and exit (for debug)",
    ),
    "2lenses": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="Toggle to indicate I5S data",
        default_value=0,
    ),
    "bessel": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description="Toggle to indicate Bessel-SIM data",
        default_value=0,
    ),
    "besselExWave": SettingFormat(
        SettingConverters.FLOAT,
        description="Bessel SIM excitation wavelength in microns",
        default_value=Decimal("0.488"),
        depends_on=(
            ConditionalSettingState("bessel", comp=Comparators.EQUAL, value=1),
        ),
    ),
    "besselNA": SettingFormat(
        SettingConverters.FLOAT,
        description="Bessel SIM excitation NA",
        default_value=Decimal("0.144"),
        depends_on=(
            ConditionalSettingState("bessel", comp=Comparators.EQUAL, value=1),
        ),
    ),
    "deskew": SettingFormat(
        SettingConverters.FLOAT,
        description="Deskew angle; if not 0.0 then perform deskewing before processing",
        default_value=Decimal(0),
    ),
    "deskewshift": SettingFormat(
        SettingConverters.INT_POSITIVE,
        description="If deskewed, shift the output image by this in X (positive->left)",
        default_value=0,
        depends_on=(
            ConditionalSettingState("deskew", comp=Comparators.NOT_EQUAL, value=0),
        ),
    ),
    "noRecon": SettingFormat(
        SettingConverters.INT_FROM_BOOL,
        nargs=0,
        description='No reconstruction will be performed; useful when combined with "deskew"',
        default_value=0,
    ),
    "cropXY": SettingFormat(
        SettingConverters.INT_POSITIVE,
        description="Crop the X-Y dimension to this number; 0 means no cropping",
        default_value=0,
    ),
    "xyres": SettingFormat(
        SettingConverters.FLOAT,
        description="X-Y pixel size (use metadata value by default)",
        # default_value=Decimal("0.1"),
    ),
    "zres": SettingFormat(
        SettingConverters.FLOAT,
        description="Z pixel size (use metadata value by default)",
        # default_value=Decimal("0.2"),
    ),
    "zresPSF": SettingFormat(
        SettingConverters.FLOAT,
        description='Z pixel size of PSF (use "zres" value by default)',
        # default_value=Decimal("0.15"),
    ),
    "wavelength": SettingFormat(
        SettingConverters.INT_GREATER_THAN_ONE,
        description="Emission wavelength in nm (use metadata value by default)",
        # default_value=530,
    ),
    # Not valid without cudasirecon's MRC/DV handling:
    # "writeTitle": SettingFormat(
    #     SettingConverters.INT_FROM_BOOL,
    #     nargs=0,
    #     description="Write command line to MRC/DV header (may cause issues with bioformats)",
    # ),
}


def filter_out_invalid_kwargs(
    kwargs: dict[str, Any],
    formatters: dict[str, SettingFormat],
    allow_none: bool = True,
) -> dict[str, Any]:
    output_kwargs = kwargs.copy()
    for arg_name, value in kwargs.items():
        setting_format = formatters.get(arg_name)
        if (
            (not allow_none and value is None)  # Check for None values
            or setting_format is None  # Check setting is in formatters
            or (
                # Check if any dependencies are unmet
                setting_format.depends_on is not None
                and any(
                    not dep.check_condition_met(output_kwargs.get(dep.name))
                    for dep in setting_format.depends_on
                )
            )
        ):
            del output_kwargs[arg_name]

    return output_kwargs


def formatters_to_default_value_kwargs(
    formatters: dict[str, SettingFormat]
) -> dict[str, Any]:
    return {
        setting_name: setting_format.default_value
        for setting_name, setting_format in formatters.items()
        if setting_format.default_value is not None
    }
