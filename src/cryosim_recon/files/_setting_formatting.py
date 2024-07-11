from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Callable


@dataclass
class SettingFormat:
    conv: Callable[[str], Any]
    split: bool = False


FORMATTERS: dict[str, SettingFormat] = {
    "ndirs": SettingFormat(int),  # number of SIM  directions
    "nphases": SettingFormat(int),  # number of pattern phases per SIM direction
    "nordersout": SettingFormat(
        int
    ),  # number of output SIM orders (must be <= nphases//2; safe to ignore usually)
    "angle0": SettingFormat(Decimal),  # angle of the first SIM angle in radians
    "ls": SettingFormat(Decimal),  # line spacing of SIM pattern in microns
    "na": SettingFormat(Decimal),  # detection objective's numerical aperture
    "nimm": SettingFormat(Decimal),  # refractive index of immersion medium
    "wiener": SettingFormat(
        float
    ),  # Wiener constant; lower value leads to higher resolution and noise;
    # playing with it extensively is strongly encouraged
    "zoomfact": SettingFormat(
        int
    ),  # lateral zoom factor in the output over the input images;
    # leaving it at 2 should be fine in most cases
    "zzoom": SettingFormat(int),  # axial zoom factor; almost never needed
    "background": SettingFormat(int),  # camera readout background
    "usecorr": SettingFormat(Path),  # use a flat-field correction file if provided
    "k0angles": SettingFormat(
        float, True
    ),  # user these pattern vector k0 angles for all
    # directions (instead of inferring the rest agnles
    # from angle0)
    "otfRA": SettingFormat(int),  # using rotationally averaged OTF; otherwise using
    # 3/2D OTF for 3/2D raw data
    "otfPerAngle": SettingFormat(
        int
    ),  # using one OTF per SIM angle; otherwise one OTF is
    # used for all angles, which is how it's been done
    # traditionally
    "fastSI": SettingFormat(int),  # SIM image is organized in Z->Angle->Phase order;
    # otherwise assuming Angle->Z->Phase image order
    "k0searchAll": SettingFormat(int),  # search for k0 at all time points
    "norescale": SettingFormat(int),  # no bleach correction
    "equalizez": SettingFormat(int),  # bleach correction for z
    "equalizet": SettingFormat(int),  # bleach correction for time
    "dampenOrder0": SettingFormat(
        int
    ),  # dampen order-0 in final assembly; do not use for 2D SIM; good choice for high-background images
    "nosuppress": SettingFormat(
        int
    ),  # do not suppress DC singularity in the result (good choice for 2D/TIRF data)
    "nokz0": SettingFormat(
        int
    ),  # not using kz=0 plane of the 0th order in the final assembly (mostly for debug)
    "gammaApo": SettingFormat(int),  # output apodization gamma; 1.0 means triangular
    # apo; lower value means less dampening of
    # high-resolution info at the tradeoff of higher
    # noise
    "explodefact": SettingFormat(int),  # artificially exploding the reciprocal-space
    # distance between orders by this factor (for debug)
    "nofilterovlps": SettingFormat(int),
    "deskew": SettingFormat(
        float
    ),  # Deskew angle; if not 0.0 then perform deskewing before processing
    "deskewshift": SettingFormat(
        int
    ),  # If deskewed, the output image's extra shift in X (positive->left)
    "noRecon": SettingFormat(
        int
    ),  # No reconstruction will be performed; useful when combined with "deskew":
    "cropXY": SettingFormat(
        int
    ),  # # Crop the X-Y dimension to this number; 0 means no cropping
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
    "xyres": SettingFormat(Decimal),  # x-y pixel size (only used for TIFF files)
    "zres": SettingFormat(Decimal),  # z pixel size (only used for TIFF files)
    "zresPSF": SettingFormat(Decimal),  # z pixel size used in PSF TIFF files)
    "wavelength": SettingFormat(
        int
    ),  # emission wavelength in nanometers (only used for TIFF files)
    #
    # OTF specific args:
    "beaddiam": SettingFormat(
        Decimal
    ),  # The diameter of the bead in microns, by default 0.12
    "angle": SettingFormat(
        Decimal
    ),  # The k0 vector angle with which the PSF is taken, by default 0
    "nocompen": SettingFormat(
        bool
    ),  # Do not perform bead size compensation, default False (do perform)
    "fixorigin": SettingFormat(
        int, True
    ),  # The starting and end pixel for interpolation along kr axis, by default (2, 9)
    "leavekz": SettingFormat(
        int, True
    ),  # Pixels to be retained on kz axis, by default (0, 0, 0)
}
