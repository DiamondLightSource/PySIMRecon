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

### Installation

##### Requirements

- This package requires Conda for package management. This project recommends using [miniforge](https://conda-forge.org/download/) from conda-forge.
- This package required a CUDA-compatible NVIDIA GPU:
  - At the time of writing, the latest version of cudasirecon (1.2.0) requires CUDA 11.8+. Older versions of cudasirecon can work with older GPUs (10.2+ is supported) but this is not recommended.
  - The CUDA version that cudasirecon is built for is dependent on conda-forge's support.
  - It is recommended to use the latest GPU drivers available for you system. Please see [cudasirecon's README](https://github.com/scopetools/cudasirecon/blob/main/README.md#gpu-requirements) for more details about CUDA and driver versions.
- Unfortunately, macOS is not supported.

##### Steps

This is not yet published on conda-forge, so the installation process is fairly manual.

1. Within a conda terminal, create a new environment `conda create -n pysimrecon_env python=3.12` and activate it `conda activate pysimrecon_env`. The environment has been called `pysimrecon_env` in this example but this is not a requirement.
2. Clone (or download) this repository.
3. Navigate to where you've cloned the repo within your terminal.
4. Install the requirements from the conda_requirements.txt file `conda install --file conda_requirements.txt`. If there are any issues with this step, please refer to the [requirements](#requirements) section.
5. Install the package with pip, now that the requirements have been met `python -m pip install .` The argument `-e` can be added if you want it install it in editable mode.

If you have any problems installing this package, please open an issue.

### Configuration
Calls to `sim-otf` and `sim-recon` can both take a `-c`/`--config` argument. This should be similar to the `config.ini` file in the configs directory, which specifies the defaults config, any per-channel configs (some values should be set on a per-channel basis), and the locations of OTFs for each channel.
The channels are specified based on the emission wavelength in nanometres (must be an integer).
##### Example:
For a channel with an emission wavelength of 525nm:
- In the `[configs]` section, the channel config can be specified as `525=/path/to/configs/525.cfg`
- In the `[otfs]` section, the channel OTF file can be specified as `525=/path/to/otfs/525_otf.tiff`
A directory can be specified within each of the `[configs]` and `[otfs]` sections. If given, files can also be defined relative to that directory, e.g. if `directory=/path/to`, then `525=/path/to/otfs/525_otf.tiff` could simply be `525=otfs/525_otf.tiff`.
Config settings will be overriden by higher-priority sources.

##### Order of setting priority:
1. Command line arguments (override all)
2. Per-channel configs (override defaults for the appropriate channel only)
3. Defaults (override any cudasirecon defaults)

### CLI

##### sim-otf

```
usage: sim-otf [-c CONFIG_PATH] [-p PSF_PATHS [PSF_PATHS ...]]
               [-o OUTPUT_DIRECTORY] [--overwrite] [--no-cleanup]
               [--shape XY_SHAPE XY_SHAPE] [-v] [--no-progress]
               [--nphases NPHASES] [--ls LS] [--na NA] [--nimm NIMM]
               [--background BACKGROUND] [--beaddiam BEADDIAM] [--angle ANGLE]
               [--nocompen NOCOMPEN] [--5bands]
               [--fixorigin FIXORIGIN FIXORIGIN]
               [--leavekz LEAVEKZ LEAVEKZ LEAVEKZ] [--I2M I2M] [-h]

SIM PSFs to OTFs

options:
  -c CONFIG_PATH, --config-path CONFIG_PATH
                        Path to the root config that specifies the paths to
                        the OTFs and the other configs
  -p PSF_PATHS [PSF_PATHS ...], --psf PSF_PATHS [PSF_PATHS ...]
                        Paths to PSF files to be reconstructed (multiple paths
                        can be given)
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        If specified, the output directory that the OTFs will
                        be saved in, otherwise each OTF will be saved in the
                        same directory as its PSF
  --overwrite           If specified, files will be overwritten if they
                        already exist (unique filenames will be used
                        otherwise)
  --no-cleanup          If specified, files created during the OTF creation
                        process will not be cleaned up
  --shape XY_SHAPE XY_SHAPE
                        Takes 2 integers (X Y), specifying the shape to crop
                        PSFs to before converting (powers of 2 are fastest)
  -v, --verbose         Show more logging
  --no-progress         turn off progress bars (only has an effect if tqdm is
                        installed)
  -h, --help            show this help message and exit

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
  --nocompen NOCOMPEN   Do not perform bead size compensation, default False
                        (do perform)
  --5bands              Output to 5 OTF bands (default is combining higher-
                        order's real and imag bands into one output
  --fixorigin FIXORIGIN FIXORIGIN
                        The starting and end pixel for interpolation along kr
                        axis (default=(2, 9))
  --leavekz LEAVEKZ LEAVEKZ LEAVEKZ
                        Pixels to be retained on kz axis (default=(0, 0, 0))
  --I2M I2M             I2M OTF file (input data contains I2M PSF)
```

##### sim-recon

```
usage: sim-recon [-c CONFIG_PATH] [-d SIM_DATA_PATHS [SIM_DATA_PATHS ...]]
                 [-o OUTPUT_DIRECTORY] [--overwrite] [--no-cleanup]
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
                 [--zresPSF ZRESPSF] [--wavelength WAVELENGTH] [-h]

Reconstruct SIM data

options:
  -c CONFIG_PATH, --config-path CONFIG_PATH
                        Path to the root config that specifies the paths to
                        the OTFs and the other configs
  -d SIM_DATA_PATHS [SIM_DATA_PATHS ...], --data SIM_DATA_PATHS [SIM_DATA_PATHS ...]
                        Paths to SIM data files to be reconstructed (multiple
                        paths can be given)
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        The output directory to save reconstructed files in
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
  -h, --help            show this help message and exit

Overrides:
  Arguments that override configured values. Defaults stated are only used
  if no value is given or configured.

  --ndirs NDIRS         Number of SIM directions (default=3)
  --nphases NPHASES     Number of pattern phases per SIM direction (default=5)
  --nordersout NORDERSOUT
                        Number of output SIM orders (must be <= nphases//2;
                        safe to ignore usually) (default=0)
  --angle0 ANGLE0       Angle of the first SIM angle in radians
                        (default=1.648)
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
  --otfRA               Use rotationally averaged OTF; otherwise use 3/2D OTF
                        for 3/2D raw data
  --otfPerAngle         Use one OTF per SIM angle; otherwise one OTF is used
                        for all angles, which is how it's been done
                        traditionally
  --fastSI              SIM image is organized in Z->Angle->Phase order;
                        otherwise assume Angle->Z->Phase image order
  --k0searchAll K0SEARCHALL
                        Search for k0 at all time points
  --norescale NORESCALE
                        No bleach correction
  --equalizez           Bleach correction for z
  --equalizet           Bleach correction for time
  --dampenOrder0        Dampen order-0 in final assembly; do not use for 2D
                        SIM; good choice for high-background images
  --nosuppress NOSUPPRESS
                        Do not suppress DC singularity in the result (good
                        choice for 2D/TIRF data)
  --nokz0               Do not use kz=0 plane of the 0th order in the final
                        assembly (mostly for debug)
  --gammaApo GAMMAAPO   Output apodization gamma; 1.0 means triangular apo;
                        lower value means less dampening of high-resolution
                        info at the trade-off of higher noise (default=1)
  --explodefact EXPLODEFACT
                        Artificially explode the reciprocal-space distance
                        between orders by this factor (for debug) (default=1)
  --nofilterovlps NOFILTEROVLPS
                        Do not filter the overlapping region between bands
                        (for debug) (default=False)
  --saveprefiltered SAVEPREFILTERED
                        Save separated bands (half Fourier space) into a file
                        and exit (for debug)
  --savealignedraw SAVEALIGNEDRAW
                        Save drift-fixed raw data (half Fourier space) into a
                        file and exit (for debug)
  --saveoverlaps SAVEOVERLAPS
                        Save overlap0 and overlap1 (real-space complex data)
                        into a file and exit (for debug)
  --2lenses             Toggle to indicate I5S data
  --bessel              Toggle to indicate Bessel-SIM data
  --besselExWave BESSELEXWAVE
                        Bessel SIM excitation wavelength in microns
                        (default=0.488)
  --besselNA BESSELNA   Bessel SIM excitation NA (default=0.144)
  --deskew DESKEW       Deskew angle; if not 0.0 then perform deskewing before
                        processing (default=0)
  --deskewshift DESKEWSHIFT
                        If deskewed, shift the output image by this in X
                        (positive->left) (default=0)
  --noRecon             No reconstruction will be performed; useful when
                        combined with "deskew"
  --cropXY CROPXY       Crop the X-Y dimension to this number; 0 means no
                        cropping (default=0)
  --xyres XYRES         X-Y pixel size (use metadata value by default)
  --zres ZRES           Z pixel size (use metadata value by default)
  --zresPSF ZRESPSF     Z pixel size of PSF (use "zres" value by default)
  --wavelength WAVELENGTH
                        Emission wavelength in nm (use metadata value by
                        default)
```

##### otf-view

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

##### dv2tiff

Accepts a list of DV files to be converted to TIFFs (with some OME metadata).
