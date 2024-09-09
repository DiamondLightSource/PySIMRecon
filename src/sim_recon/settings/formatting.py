from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal
from configparser import RawConfigParser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Callable, Literal


# Using the logic from RawConfigParser for consistency
__boolean_conv = RawConfigParser()._convert_to_boolean  # type: ignore[attr-defined]


@dataclass
class SettingFormat:
    conv: Callable[[str], Any]
    nargs: int | Literal["+"] = 1  # "+" matching argparse
    description: str | None = None
    default_value: Any = None

    @property
    def help_string(self) -> str | None:
        help_string_parts = []
        if self.description is not None:
            help_string_parts.append(self.description)
        if self.default_value is not None:
            help_string_parts.append(f"(default={self.default_value})")
        if help_string_parts:
            return " ".join(help_string_parts)
        return None


OTF_FORMATTERS: dict[str, SettingFormat] = {
    # Shared arguments:
    "nphases": SettingFormat(
        int, description="Number of pattern phases per SIM direction", default_value=5
    ),
    "ls": SettingFormat(
        Decimal,
        description="Line spacing of SIM pattern in microns",
        default_value=Decimal("0.172"),
    ),
    "na": SettingFormat(
        Decimal,
        description="Detection objective's numerical aperture",
        default_value=Decimal("1.2"),
    ),
    "nimm": SettingFormat(
        Decimal,
        description="Refractive index of immersion medium",
        default_value=Decimal("1.33"),
    ),
    "background": SettingFormat(
        Decimal, description="Camera readout background", default_value=Decimal(0)
    ),
    # OTF specific args:
    "beaddiam": SettingFormat(
        Decimal,
        description="The diameter of the bead in microns",
        default_value=Decimal("0.12"),
    ),
    "angle": SettingFormat(
        Decimal,
        description="The k0 vector angle with which the PSF is taken",
        default_value=Decimal(0),
    ),
    "nocompen": SettingFormat(
        __boolean_conv,
        description="Do not perform bead size compensation, default False (do perform)",
    ),
    "5bands": SettingFormat(
        bool,
        description="Output to 5 OTF bands (default is combining higher-order's real and imag bands into one output",
    ),
    "fixorigin": SettingFormat(
        int,
        2,
        description="The starting and end pixel for interpolation along kr axis",
        default_value=(2, 9),
    ),
    "leavekz": SettingFormat(
        int, 3, description="Pixels to be retained on kz axis", default_value=(0, 0, 0)
    ),
    "I2M": SettingFormat(
        Path, description="I2M OTF file (input data contains I2M PSF)"
    ),
}


RECON_FORMATTERS: dict[str, SettingFormat] = {
    "ndirs": SettingFormat(
        int, description="Number of SIM  directions", default_value=3
    ),
    "nphases": SettingFormat(
        int, description="Number of pattern phases per SIM direction", default_value=5
    ),
    "nordersout": SettingFormat(
        int,
        description="Number of output SIM orders (must be <= nphases//2; safe to ignore usually)",
        default_value=0,
    ),
    "angle0": SettingFormat(
        Decimal,
        description="Angle of the first SIM angle in radians",
        default_value=Decimal("1.648"),
    ),
    "ls": SettingFormat(
        Decimal,
        description="Line spacing of SIM pattern in microns",
        default_value=Decimal("0.172"),
    ),
    "na": SettingFormat(
        Decimal,
        description="Detection objective's numerical aperture",
        default_value=Decimal("1.2"),
    ),
    "nimm": SettingFormat(
        Decimal,
        description="Refractive index of immersion medium",
        default_value=Decimal("1.33"),
    ),
    "wiener": SettingFormat(
        Decimal,
        description="Wiener constant; lower value leads to higher resolution and noise (playing with it extensively is strongly encouraged)",
        default_value=Decimal("0.01"),
    ),
    "otfcutoff": SettingFormat(
        Decimal,
        description='OTF threshold below which it\'ll be considered noise and not used in "makeoverlaps"',
        default_value=Decimal("0.006"),
    ),
    "zoomfact": SettingFormat(
        Decimal,
        description="Lateral zoom factor in the output over the input images",
        default_value=Decimal(2),
    ),
    "zzoom": SettingFormat(int, description="Axial zoom factor", default_value=1),
    "background": SettingFormat(
        Decimal, description="Camera readout background", default_value=Decimal(0)
    ),
    "usecorr": SettingFormat(
        Path, description="Use a flat-field correction file if provided"
    ),
    "forcemodamp": SettingFormat(
        Decimal,
        "+",
        description="Force modamps to these values; useful when image quality is low and auto-estimated modamps are below, say, 0.1",
    ),
    "k0angles": SettingFormat(
        Decimal,
        3,
        description="Use these pattern vector k0 angles for all directions (instead of inferring the rest angles from angle0)",
    ),
    "otfRA": SettingFormat(
        bool,
        description="Use rotationally averaged OTF; otherwise use 3/2D OTF for 3/2D raw data",
    ),
    "otfPerAngle": SettingFormat(
        bool,
        description="Use one OTF per SIM angle; otherwise one OTF is used for all angles, which is how it's been done traditionally",
    ),
    "fastSI": SettingFormat(
        bool,
        description="SIM image is organized in Z->Angle->Phase order; otherwise assume Angle->Z->Phase image order",
    ),
    "k0searchAll": SettingFormat(int, description="Search for k0 at all time points"),
    "norescale": SettingFormat(int, description="No bleach correction"),
    "equalizez": SettingFormat(bool, description="Bleach correction for z"),
    "equalizet": SettingFormat(bool, description="Bleach correction for time"),
    "dampenOrder0": SettingFormat(
        bool,
        description="Dampen order-0 in final assembly; do not use for 2D SIM; good choice for high-background images",
    ),
    "nosuppress": SettingFormat(
        int,
        description="Do not suppress DC singularity in the result (good choice for 2D/TIRF data)",
    ),
    "nokz0": SettingFormat(
        bool,
        description="Do not use kz=0 plane of the 0th order in the final assembly (mostly for debug)",
    ),
    "gammaApo": SettingFormat(
        Decimal,
        description="Output apodization gamma; 1.0 means triangular apo; lower value means less dampening of high-resolution info at the trade-off of higher noise",
        default_value=Decimal(1),
    ),
    "explodefact": SettingFormat(
        Decimal,
        description="Artificially explode the reciprocal-space distance between orders by this factor (for debug)",
        default_value=Decimal(1),
    ),
    "nofilterovlps": SettingFormat(
        __boolean_conv,
        description="Do not filter the overlapping region between bands (for debug)",
        default_value=False,
    ),
    "saveprefiltered": SettingFormat(
        Path,
        description="Save separated bands (half Fourier space) into a file and exit (for debug)",
    ),
    "savealignedraw": SettingFormat(
        Path,
        description="Save drift-fixed raw data (half Fourier space) into a file and exit (for debug)",
    ),
    "saveoverlaps": SettingFormat(
        Path,
        description="Save overlap0 and overlap1 (real-space complex data) into a file and exit (for debug)",
    ),
    "2lenses": SettingFormat(bool, description="Toggle to indicate I5S data"),
    "bessel": SettingFormat(bool, description="Toggle to indicate Bessel-SIM data"),
    "besselExWave": SettingFormat(
        Decimal,
        description="Bessel SIM excitation wavelength in microns",
        default_value=Decimal("0.488"),
    ),
    "besselNA": SettingFormat(
        Decimal, description="Bessel SIM excitation NA", default_value=Decimal("0.144")
    ),
    "deskew": SettingFormat(
        Decimal,
        description="Deskew angle; if not 0.0 then perform deskewing before processing",
        default_value=Decimal(0),
    ),
    "deskewshift": SettingFormat(
        int,
        description="If deskewed, shift the output image by this in X (positive->left)",
        default_value=0,
    ),
    "noRecon": SettingFormat(
        bool,
        description='No reconstruction will be performed; useful when combined with "deskew"',
    ),
    "cropXY": SettingFormat(
        int,
        description="Crop the X-Y dimension to this number; 0 means no cropping",
        default_value=0,
    ),
    "xyres": SettingFormat(
        Decimal,
        description="X-Y pixel size (use metadata value by default)",
        # default_value=Decimal("0.1"),
    ),
    "zres": SettingFormat(
        Decimal,
        description="Z pixel size (use metadata value by default)",
        # default_value=Decimal("0.2"),
    ),
    "zresPSF": SettingFormat(
        Decimal,
        description='Z pixel size of PSF (use "zres" value by default)',
        # default_value=Decimal("0.15"),
    ),
    "wavelength": SettingFormat(
        int,
        description="Emission wavelength in nm (use metadata value by default)",
        # default_value=530,
    ),
    # Not valid without cudasirecon's MRC/DV handling:
    # "writeTitle": SettingFormat(
    #     bool,
    #     description="Write command line to MRC/DV header (may cause issues with bioformats)",
    # ),
}
