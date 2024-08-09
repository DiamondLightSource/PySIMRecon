from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal
from configparser import RawConfigParser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Callable


# Using the logic from RawConfigParser for consistency
__boolean_conv = RawConfigParser()._convert_to_boolean


@dataclass
class SettingFormat:
    conv: Callable[[str], Any]
    count: int = 1
    description: str | None = None


OTF_FORMATTERS: dict[str, SettingFormat] = {
    # Shared arguments:
    "nphases": SettingFormat(
        int, description="number of pattern phases per SIM direction"
    ),
    "ls": SettingFormat(Decimal, description="line spacing of SIM pattern in microns"),
    "na": SettingFormat(
        Decimal, description="detection objective's numerical aperture"
    ),
    "nimm": SettingFormat(Decimal, description="refractive index of immersion medium"),
    "background": SettingFormat(int, description="camera readout background"),
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
    "ndirs": SettingFormat(int, description="number of SIM  directions"),
    "nphases": SettingFormat(
        int, description="number of pattern phases per SIM direction"
    ),
    "nordersout": SettingFormat(
        int,
        description="number of output SIM orders (must be <= nphases//2; safe to ignore usually)",
    ),
    "angle0": SettingFormat(
        Decimal, description="angle of the first SIM angle in radians"
    ),
    "ls": SettingFormat(Decimal, description="line spacing of SIM pattern in microns"),
    "na": SettingFormat(
        Decimal, description="detection objective's numerical aperture"
    ),
    "nimm": SettingFormat(Decimal, description="refractive index of immersion medium"),
    "wiener": SettingFormat(
        Decimal,
        description="Wiener constant; lower value leads to higher resolution and noise;",
    ),
    # playing with it extensively is strongly encouraged
    "zoomfact": SettingFormat(
        Decimal, description="lateral zoom factor in the output over the input images;"
    ),
    # leaving it at 2 should be fine in most cases
    "zzoom": SettingFormat(int, description="axial zoom factor; almost never needed"),
    "background": SettingFormat(int, description="camera readout background"),
    "usecorr": SettingFormat(
        Path, description="use a flat-field correction file if provided"
    ),
    "k0angles": SettingFormat(
        Decimal, 3, description="user these pattern vector k0 angles for all"
    ),
    # directions (instead of inferring the rest agnles
    # from angle0)
    "otfRA": SettingFormat(
        int, description="using rotationally averaged OTF; otherwise using"
    ),
    # 3/2D OTF for 3/2D raw data
    "otfPerAngle": SettingFormat(
        int, description="using one OTF per SIM angle; otherwise one OTF is"
    ),
    # used for all angles, which is how it's been done
    # traditionally
    "fastSI": SettingFormat(
        int, description="SIM image is organized in Z->Angle->Phase order;"
    ),
    # otherwise assuming Angle->Z->Phase image order
    "k0searchAll": SettingFormat(int, description="search for k0 at all time points"),
    "norescale": SettingFormat(int, description="no bleach correction"),
    "equalizez": SettingFormat(int, description="bleach correction for z"),
    "equalizet": SettingFormat(int, description="bleach correction for time"),
    "dampenOrder0": SettingFormat(
        int,
        description="dampen order-0 in final assembly; do not use for 2D SIM; good choice for high-background images",
    ),
    "nosuppress": SettingFormat(
        int,
        description="do not suppress DC singularity in the result (good choice for 2D/TIRF data)",
    ),
    "nokz0": SettingFormat(
        int,
        description="not using kz=0 plane of the 0th order in the final assembly (mostly for debug)",
    ),
    "gammaApo": SettingFormat(
        Decimal, description="output apodization gamma; 1.0 means triangular"
    ),
    # apo; lower value means less dampening of
    # high-resolution info at the tradeoff of higher
    # noise
    "explodefact": SettingFormat(
        int, description="artificially exploding the reciprocal-space"
    ),
    # distance between orders by this factor (for debug)
    "nofilterovlps": SettingFormat(int),
    "deskew": SettingFormat(
        Decimal,
        description="Deskew angle; if not 0.0 then perform deskewing before processing",
    ),
    "deskewshift": SettingFormat(
        int,
        description="If deskewed, the output image's extra shift in X (positive->left)",
    ),
    "noRecon": SettingFormat(
        int,
        description='No reconstruction will be performed; useful when combined with "deskew":',
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
    #
    # These args are taken from the MRC file so will be ignored:
    "xyres": SettingFormat(
        Decimal, description="x-y pixel size (only used for TIFF files)"
    ),
    "zres": SettingFormat(
        Decimal, description="z pixel size (only used for TIFF files)"
    ),
    "zresPSF": SettingFormat(
        Decimal, description="z pixel size (used in PSF TIFF files)"
    ),
    "wavelength": SettingFormat(
        int, description="emission wavelength in nanometers (only used for TIFF files)"
    ),
}
