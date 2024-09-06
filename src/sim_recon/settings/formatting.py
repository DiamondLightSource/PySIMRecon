from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal
from configparser import RawConfigParser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Callable


# Using the logic from RawConfigParser for consistency
__boolean_conv = RawConfigParser()._convert_to_boolean  # type: ignore[attr-defined]


@dataclass
class SettingFormat:
    conv: Callable[[str], Any]
    count: int = 1
    description: str | None = None


OTF_FORMATTERS: dict[str, SettingFormat] = {
    # Shared arguments:
    "nphases": SettingFormat(
        int, description="Number of pattern phases per SIM direction"
    ),
    "ls": SettingFormat(Decimal, description="Line spacing of SIM pattern in microns"),
    "na": SettingFormat(
        Decimal, description="Detection objective's numerical aperture"
    ),
    "nimm": SettingFormat(Decimal, description="Refractive index of immersion medium"),
    "background": SettingFormat(int, description="Camera readout background"),
    # OTF specific args:
    "beaddiam": SettingFormat(
        Decimal, description="The diameter of the bead in microns, by default 0.12"
    ),
    "angle": SettingFormat(
        Decimal,
        description="The k0 vector angle with which the PSF is taken, by default 0",
    ),
    "nocompen": SettingFormat(
        __boolean_conv,
        description="Do not perform bead size compensation, default False (do perform)",
    ),
    "fixorigin": SettingFormat(
        int,
        2,
        description="The starting and end pixel for interpolation along kr axis, by default (2, 9)",
    ),
    "leavekz": SettingFormat(
        int, 3, description="Pixels to be retained on kz axis, by default (0, 0, 0)"
    ),
}

RECON_FORMATTERS: dict[str, SettingFormat] = {
    "ndirs": SettingFormat(int, description="Number of SIM  directions"),
    "nphases": SettingFormat(
        int, description="Number of pattern phases per SIM direction"
    ),
    "nordersout": SettingFormat(
        int,
        description="Number of output SIM orders (must be <= nphases//2; safe to ignore usually)",
    ),
    "angle0": SettingFormat(
        Decimal, description="Angle of the first SIM angle in radians"
    ),
    "ls": SettingFormat(Decimal, description="Line spacing of SIM pattern in microns"),
    "na": SettingFormat(
        Decimal, description="Detection objective's numerical aperture"
    ),
    "nimm": SettingFormat(Decimal, description="Refractive index of immersion medium"),
    "wiener": SettingFormat(
        Decimal,
        description="Wiener constant; lower value leads to higher resolution and noise; playing with it extensively is strongly encouraged",
    ),
    "zoomfact": SettingFormat(
        Decimal,
        description="Lateral zoom factor in the output over the input images; leaving it at 2 should be fine in most cases",
    ),
    "zzoom": SettingFormat(int, description="Axial zoom factor; almost never needed"),
    "background": SettingFormat(int, description="Camera readout background"),
    "usecorr": SettingFormat(
        Path, description="Use a flat-field correction file if provided"
    ),
    "k0angles": SettingFormat(
        Decimal,
        3,
        description="Use these pattern vector k0 angles for all directions (instead of inferring the rest angles from angle0)",
    ),
    "otfRA": SettingFormat(
        int,
        description="Use rotationally averaged OTF; otherwise use 3/2D OTF for 3/2D raw data",
    ),
    "otfPerAngle": SettingFormat(
        int,
        description="Use one OTF per SIM angle; otherwise one OTF is used for all angles, which is how it's been done traditionally",
    ),
    "fastSI": SettingFormat(
        int,
        description="SIM image is organized in Z->Angle->Phase order; otherwise assume Angle->Z->Phase image order",
    ),
    "k0searchAll": SettingFormat(int, description="Search for k0 at all time points"),
    "norescale": SettingFormat(int, description="No bleach correction"),
    "equalizez": SettingFormat(int, description="Bleach correction for z"),
    "equalizet": SettingFormat(int, description="Bleach correction for time"),
    "dampenOrder0": SettingFormat(
        int,
        description="Dampen order-0 in final assembly; do not use for 2D SIM; good choice for high-background images",
    ),
    "nosuppress": SettingFormat(
        int,
        description="Do not suppress DC singularity in the result (good choice for 2D/TIRF data)",
    ),
    "nokz0": SettingFormat(
        int,
        description="Do not use kz=0 plane of the 0th order in the final assembly (mostly for debug)",
    ),
    "gammaApo": SettingFormat(
        Decimal,
        description="Output apodization gamma; 1.0 means triangular apo; lower value means less dampening of high-resolution info at the trade-off of higher noise",
    ),
    "explodefact": SettingFormat(
        int,
        description="Artificially explode the reciprocal-space distance between orders by this factor (for debug)",
    ),
    "nofilterovlps": SettingFormat(
        int,
        description="Do not filter the overlapping region between bands (for debug)",
    ),
    "deskew": SettingFormat(
        Decimal,
        description="Deskew angle; if not 0.0 then perform deskewing before processing",
    ),
    "deskewshift": SettingFormat(
        int,
        description="If deskewed, shift the output image by this in X (positive->left)",
    ),
    "noRecon": SettingFormat(
        int,
        description='No reconstruction will be performed; useful when combined with "deskew"',
    ),
    "cropXY": SettingFormat(
        int, description="Crop the X-Y dimension to this number; 0 means no cropping"
    ),
    #
    # These args are not used as it is not clear what they take
    # "forcemodamp": arg           modamps to be forced to these values; useful when
    # image quality is low and auto-estimated modamps
    # are below, say, 0.1
    # not filtering the overlaping region between bands (for debug)
    # "saveprefiltered": arg       save separated bands (half Fourier space) into a file and exit (for debug)
    # "savealignedraw": arg        save drift-fixed raw data (half Fourier space) into a file and exit (for debug)
    # "saveoverlaps": arg          save overlap0 and overlap1 (real-space complex data) into a file and exit (for debug)
    # "2lenses":                   I5S data
    # "bessel":                    bessel-SIM data
    # "besselExWave": arg (=0.488) Bessel SIM excitation wavelength in microns
    # "besselNA": arg (=0.144)     Bessel SIM excitation NA)
    "xyres": SettingFormat(
        Decimal, description="X-Y pixel size (use metadata value by default)"
    ),
    "zres": SettingFormat(
        Decimal, description="Z pixel size (use metadata value by default)"
    ),
    "zresPSF": SettingFormat(
        Decimal, description="Z pixel size of PSF (use zres value by default)"
    ),
    "wavelength": SettingFormat(
        int, description="Emission wavelength in nm (use metadata value by default)"
    ),
}
