# PySIMRecon

Easy to use wrapper for [pycudasirecon](https://github.com/tlambert03/pycudasirecon)/[cudasirecon](https://github.com/scopetools/cudasirecon), allowing the use of DV (DeltaVision) files without IVE/Priism (UCSF library with MRC/DV support).

This is built for easy use from the command line or as part of an auto-processing pipeline, allowing parameters to be pre-configured or defined at runtime.

### Goal

To reconstruct the SIM data and create OTFs from PSFs from the cryoSIM at B24, Diamond from the .dv files created by Cockpit. Requirements are:

- Easy to use
- Open source
- Reproducible results
- Can be used as part of an automatic processing pipeline

### Current state

Under development

## Installation

#### Requirements

- This package requires Conda for package management. This project recommends using [miniforge](https://conda-forge.org/download/) from conda-forge.
- This package required a CUDA-compatible NVIDIA GPU:
  - At the time of writing, the latest version of cudasirecon (1.2.0) requires CUDA 11.8+. Older versions of cudasirecon have a lower requirement of CUDA 10.2+ but are unsupported by this package.
  - The CUDA version that cudasirecon is built for is dependent on conda-forge's support.
  - It is recommended to use the latest GPU drivers available for you system. Please see [cudasirecon's README](https://github.com/scopetools/cudasirecon/blob/main/README.md#gpu-requirements) for more details about CUDA and driver versions.
- Unfortunately, macOS is not supported.

#### Steps

This is not yet published on conda-forge, so the installation process is fairly manual.

1. Within a conda terminal, create a new environment `conda create -n pysimrecon_env python=3.12` and activate it `conda activate pysimrecon_env`. The environment has been called `pysimrecon_env` in this example but this is not a requirement.
2. Clone (or download) this repository.
3. Navigate to where you've cloned the repo within your terminal.
4. Install the requirements from the conda_requirements.txt file `conda install -c conda-forge --file conda_requirements.txt`. If there are any issues with this step, please refer to the [requirements](#requirements) section.
5. Install the package with pip, now that the requirements have been met:
   * It is recommended to install the package using `python -m pip install .[progress]` as this includes progress bars using [tqdm](https://tqdm.github.io/) (simply use `python -m pip install .` if you don't want this).
   * The package can be installed in editable mode by adding the option `-e`, i.e. `python -m pip install -e .[progress]`.

If you have any problems installing this package, please open an issue.

## Configuration
Calls to `sim-otf` and `sim-recon` can both take a `-c`/`--config` argument. This should be similar to the `config.ini` file in the configs directory, which specifies the defaults config, any per-channel configs (some values should be set on a per-channel basis), and the locations of OTFs for each channel. Channels are specified based on the emission wavelength in nanometres (must be an integer).

##### Example:
For a channel with an emission wavelength of 525nm:
- In the `[configs]` section, the channel config can be specified as `525=/path/to/configs/525.cfg`
- In the `[otfs]` section, the channel OTF file can be specified as `525=/path/to/otfs/525_otf.tiff`
A directory can be specified within each of the `[configs]` and `[otfs]` sections. If given, files can also be defined relative to that directory, e.g. if `directory=/path/to`, then `525=/path/to/otfs/525_otf.tiff` could simply be `525=otfs/525_otf.tiff`.
Config settings will be overriden by higher-priority sources.

#### Configuring defaults

The defaults config and per-channel configs expect the same form, with the headers `[otf config]` for settings that are used with `sim-otf` and `[recon config]` for settings that are used with `sim-recon`. For these commands, available settings can be found in their argument form (with a leading `--` that should be removed for config use) in the [CLI](#CLI) section under 'Overrides'. While these settings can be set via command line arguments, command line arguments cannot be set per-channel and will apply to all files and channels.


#### Order of setting priority:
1. Command line arguments (override all)
2. Per-channel configs (override defaults for the appropriate channel only)
3. Defaults (override any cudasirecon defaults)

## CLI

#### sim-otf

```
usage: sim-otf [-h] -p PSF_PATHS [PSF_PATHS ...] [-c CONFIG_PATH]
               [-o OUTPUT_DIRECTORY] [--overwrite] [--no-cleanup]
               [--shape XY_SHAPE XY_SHAPE] [--centre XY_CENTRE XY_CENTRE] [-v]
               [--no-progress] [--nphases NPHASES] [--ls LS] [--na NA]
               [--nimm NIMM] [--background BACKGROUND] [--beaddiam BEADDIAM]
               [--angle ANGLE] [--nocompen] [--5bands]
               [--fixorigin FIXORIGIN FIXORIGIN]
               [--leavekz LEAVEKZ LEAVEKZ LEAVEKZ] [--I2M I2M]

SIM PSFs to OTFs

options:
  -h, --help            show this help message and exit
  -p PSF_PATHS [PSF_PATHS ...], --psf PSF_PATHS [PSF_PATHS ...]
                        Paths to PSF files to be reconstructed (multiple paths
                        can be given)
  -c CONFIG_PATH, --config-path CONFIG_PATH
                        Path to the root config that specifies the paths to
                        the OTFs and the other configs (recommended)
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        If specified, the output directory in which the OTF
                        files will be saved (otherwise each OTF will be saved
                        in the same directory as its PSF)
  --overwrite           If specified, files will be overwritten if they
                        already exist (unique filenames will be used
                        otherwise)
  --no-cleanup          If specified, files created during the OTF creation
                        process will not be cleaned up
  --shape XY_SHAPE XY_SHAPE
                        Takes 2 integers (X Y), specifying the shape to crop
                        PSFs to before converting (powers of 2 are fastest)
  --centre XY_CENTRE XY_CENTRE
                        Takes 2 floats (X Y), specifying the 0-indexed pixel
                        coordinates that PSFs are cropped around (the image
                        centre is used otherwise)
  -v, --verbose         Show more logging
  --no-progress         turn off progress bars (only has an effect if tqdm is
                        installed)

Overrides:
  Arguments that override configured values. Defaults stated are only used
  if no value is given or configured.

  --nphases NPHASES     Number of pattern phases per SIM direction (default=5)
  --ls LS               Line spacing of SIM pattern in microns (default=0.172)
  --na NA               Detection objective's numerical aperture (default=1.2)
  --nimm NIMM           Refractive index of immersion medium (default=1.33)
  --background BACKGROUND
                        Camera readout background (default=0)
  --beaddiam BEADDIAM   The diameter of the bead in microns (default=0.12)
  --angle ANGLE         The k0 vector angle with which the PSF is taken
                        (default=0)
  --nocompen            Do not perform bead size compensation
  --5bands              Output to 5 OTF bands (otherwise higher-order's real
                        and imaginary bands are combined into one output)
  --fixorigin FIXORIGIN FIXORIGIN
                        The starting and end pixel for interpolation along kr
                        axis (default=(2, 9))
  --leavekz LEAVEKZ LEAVEKZ LEAVEKZ
                        Pixels to be retained on kz axis (default=(0, 0, 0))
  --I2M I2M             I2M OTF file (input data contains I2M PSF)
```

#### sim-recon

```
usage: sim-recon [-h] -d SIM_DATA_PATHS [SIM_DATA_PATHS ...] [-c CONFIG_PATH]
                 [-o OUTPUT_DIRECTORY] [-p PROCESSING_DIRECTORY] [--otf OTFS]
                 [-amc] [--type {dv,tiff}] [--overwrite] [--no-cleanup]
                 [--keep-split] [--parallel] [-v] [--no-progress]
                 [--ndirs NDIRS] [--nphases NPHASES] [--nordersout NORDERSOUT]
                 [--angle0 ANGLE0] [--ls LS] [--na NA] [--nimm NIMM]
                 [--wiener WIENER] [--otfcutoff OTFCUTOFF]
                 [--zoomfact ZOOMFACT] [--zzoom ZZOOM]
                 [--background BACKGROUND] [--usecorr USECORR]
                 [--forcemodamp FORCEMODAMP [FORCEMODAMP ...]]
                 [--k0angles K0ANGLES K0ANGLES K0ANGLES] [--otfRA]
                 [--otfPerAngle] [--fastSI] [--k0searchAll K0SEARCHALL]
                 [--norescale NORESCALE] [--equalizez] [--equalizet]
                 [--dampenOrder0] [--nosuppress NOSUPPRESS] [--nokz0]
                 [--gammaApo GAMMAAPO] [--explodefact EXPLODEFACT]
                 [--nofilterovlps NOFILTEROVLPS]
                 [--saveprefiltered SAVEPREFILTERED]
                 [--savealignedraw SAVEALIGNEDRAW]
                 [--saveoverlaps SAVEOVERLAPS] [--2lenses] [--bessel]
                 [--besselExWave BESSELEXWAVE] [--besselNA BESSELNA]
                 [--deskew DESKEW] [--deskewshift DESKEWSHIFT] [--noRecon]
                 [--cropXY CROPXY] [--xyres XYRES] [--zres ZRES]
                 [--zresPSF ZRESPSF] [--wavelength WAVELENGTH]

Reconstruct SIM data

options:
  -h, --help            show this help message and exit
  -d SIM_DATA_PATHS [SIM_DATA_PATHS ...], --data SIM_DATA_PATHS [SIM_DATA_PATHS ...]
                        Paths to SIM data files to be reconstructed (multiple
                        paths can be given)
  -c CONFIG_PATH, --config-path CONFIG_PATH
                        Path to the root config that specifies the paths to
                        the OTFs and the other configs (recommended)
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        If specified, the output directory in which the
                        reconstructed files will be saved (otherwise each
                        reconstruction will be saved in the same directory as
                        its SIM data file)
  -p PROCESSING_DIRECTORY, --processing-directory PROCESSING_DIRECTORY
                        If specified, the directory in which the temporary
                        files will be stored for processing (otherwise the
                        output directory will be used)
  --otf OTFS            The OTF file for a channel can be specified, which
                        should be given as <emission wavelength in nm>:<the
                        path to the OTF file> e.g. '--otf
                        525:/path/to/525_otf.tiff' (argument can be given
                        multiple times to provide OTFs for multiple channels)
  -amc, --allow-missing-channels
                        If specified, attempt reconstruction of other channels
                        in a multi-channel file if one or more are not
                        configured
  --type {dv,tiff}      File type of output images
  --overwrite           If specified, files will be overwritten if they
                        already exist (unique filenames will be used
                        otherwise)
  --no-cleanup          If specified, files created during the reconstruction
                        process will not be cleaned up
  --keep-split          If specified, channels will not be stitched back
                        together after reconstruction
  --parallel            If specified, up to 2 processes will be run at a time
  -v, --verbose         Show more logging
  --no-progress         turn off progress bars (only has an effect if tqdm is
                        installed)

Overrides:
  Arguments that override configured values. Defaults stated are only used
  if no value is given or configured.

  --ndirs NDIRS         Number of SIM directions (default=3)
  --nphases NPHASES     Number of pattern phases per SIM direction (default=5)
  --nordersout NORDERSOUT
                        Number of output SIM orders (must be <= nphases//2;
                        safe to ignore usually) (default=0)
  --angle0 ANGLE0       Angle of the first SIM angle in radians
                        (default=1.648) (depends on: k0angles is None)
  --ls LS               Line spacing of SIM pattern in microns (default=0.172)
  --na NA               Detection objective's numerical aperture (default=1.2)
  --nimm NIMM           Refractive index of immersion medium (default=1.33)
  --wiener WIENER       Wiener constant; lower value leads to higher
                        resolution and noise (playing with it extensively is
                        strongly encouraged) (default=0.01)
  --otfcutoff OTFCUTOFF
                        OTF threshold below which it'll be considered noise
                        and not used in "makeoverlaps" (default=0.006)
  --zoomfact ZOOMFACT   Lateral zoom factor in the output over the input
                        images (default=2)
  --zzoom ZZOOM         Axial zoom factor (default=1)
  --background BACKGROUND
                        Camera readout background (default=0)
  --usecorr USECORR     Use a flat-field correction file if provided
  --forcemodamp FORCEMODAMP [FORCEMODAMP ...]
                        Force modamps to these values; useful when image
                        quality is low and auto-estimated modamps are below,
                        say, 0.1
  --k0angles K0ANGLES K0ANGLES K0ANGLES
                        Use these pattern vector k0 angles for all directions
                        (instead of inferring the rest angles from angle0)
  --otfRA               Use rotationally averaged OTF, otherwise 3/2D OTF is
                        used for 3/2D raw data (default=0)
  --otfPerAngle         Use one OTF per SIM angle, otherwise one OTF is used
                        for all angles, which is how it's been done
                        traditionally (default=0)
  --fastSI              SIM image is organized in Z->Angle->Phase order,
                        otherwise Angle->Z->Phase image order is assumed
                        (default=0)
  --k0searchAll K0SEARCHALL
                        Search for k0 at all time points (default=0)
  --norescale NORESCALE
                        No bleach correction (default=0)
  --equalizez           Bleach correction for z (default=0) (depends on:
                        norescale != 1)
  --equalizet           Bleach correction for time (default=0) (depends on:
                        norescale != 1)
  --dampenOrder0        Dampen order-0 in final assembly; do not use for 2D
                        SIM; good choice for high-background images
                        (default=0)
  --nosuppress NOSUPPRESS
                        Do not suppress DC singularity in the result (good
                        choice for 2D/TIRF data) (default=0)
  --nokz0               Do not use kz=0 plane of the 0th order in the final
                        assembly (mostly for debug) (default=0)
  --gammaApo GAMMAAPO   Output apodization gamma; 1.0 means triangular apo;
                        lower value means less dampening of high-resolution
                        info at the trade-off of higher noise (default=1)
  --explodefact EXPLODEFACT
                        Artificially explode the reciprocal-space distance
                        between orders by this factor (for debug) (default=1)
  --nofilterovlps NOFILTEROVLPS
                        Do not filter the overlapping region between bands
                        (for debug) (default=0)
  --saveprefiltered SAVEPREFILTERED
                        Save separated bands (half Fourier space) into a file
                        and exit (for debug)
  --savealignedraw SAVEALIGNEDRAW
                        Save drift-fixed raw data (half Fourier space) into a
                        file and exit (for debug)
  --saveoverlaps SAVEOVERLAPS
                        Save overlap0 and overlap1 (real-space complex data)
                        into a file and exit (for debug)
  --2lenses             Toggle to indicate I5S data (default=0)
  --bessel              Toggle to indicate Bessel-SIM data (default=0)
  --besselExWave BESSELEXWAVE
                        Bessel SIM excitation wavelength in microns
                        (default=0.488) (depends on: bessel == 1)
  --besselNA BESSELNA   Bessel SIM excitation NA (default=0.144) (depends on:
                        bessel == 1)
  --deskew DESKEW       Deskew angle; if not 0.0 then perform deskewing before
                        processing (default=0)
  --deskewshift DESKEWSHIFT
                        If deskewed, shift the output image by this in X
                        (positive->left) (default=0) (depends on: deskew != 0)
  --noRecon             No reconstruction will be performed; useful when
                        combined with "deskew" (default=0)
  --cropXY CROPXY       Crop the X-Y dimension to this number; 0 means no
                        cropping (default=0)
  --xyres XYRES         X-Y pixel size (use metadata value by default)
  --zres ZRES           Z pixel size (use metadata value by default)
  --zresPSF ZRESPSF     Z pixel size of PSF (use "zres" value by default)
  --wavelength WAVELENGTH
                        Emission wavelength in nm (use metadata value by
                        default)
```

#### otf-view

```
usage: otf-view [-h] [--show] [--show-only] [-o OUTPUT_DIRECTORY] [-v]
                [--no-progress]
                otf_paths [otf_paths ...]

Create OTF views

positional arguments:
  otf_paths             OTF file paths

options:
  -h, --help            show this help message and exit
  --show                Display the plots while running
  --show-only           Show plots without saving
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        Save to this directory if saving plots, otherwise each
                        plot will be saved with its input file
  -v, --verbose         Show more logging
  --no-progress         turn off progress bars (only has an effect if tqdm is
                        installed)
```

#### dv2tiff

Accepts a list of DV files to be converted to TIFFs (with some OME metadata).
