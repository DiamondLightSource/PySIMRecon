# PySIMRecon

Easy to use wrapper for pyCUDAsirecon, allowing the use of DV (DeltaVision) files without IVE/Priism (UCSF library with MRC/DV support).

This is built for easy use from the command line or as part of an auto-processing pipeline, allowing parameters to be pre-configured or defined at runtime.

### Goal

To reconstruct the SIM data and create OTFs from PSFs from the cryoSIM at B24, Diamond from the .dv files created by Cockpit. Requirements are:

- Easy to use
- Open source
- Reproducable results
- Can be used as part of an automatic processing pipeline

### Current state

Under development

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
               [--nocompen NOCOMPEN] [--fixorigin FIXORIGIN FIXORIGIN]
               [--leavekz LEAVEKZ LEAVEKZ LEAVEKZ] [-h]

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
  --no-progress         turn off progress bars (only has an affect if tqdm is
                        installed)
  -h, --help            show this help message and exit

Overrides:
  Arguments that override configured values. Defaults stated are only used
  if no value is given or configured.

  --nphases NPHASES     number of pattern phases per SIM direction
  --ls LS               line spacing of SIM pattern in microns
  --na NA               detection objective's numerical aperture
  --nimm NIMM           refractive index of immersion medium
  --background BACKGROUND
                        camera readout background
  --beaddiam BEADDIAM   The diameter of the bead in microns, by default 0.12
  --angle ANGLE         The k0 vector angle with which the PSF is taken, by
                        default 0
  --nocompen NOCOMPEN   Do not perform bead size compensation, default False
                        (do perform)
  --fixorigin FIXORIGIN FIXORIGIN
                        The starting and end pixel for interpolation along kr
                        axis, by default (2, 9)
  --leavekz LEAVEKZ LEAVEKZ LEAVEKZ
                        Pixels to be retained on kz axis, by default (0, 0, 0)
```


##### sim-recon
```
usage: sim-recon [-c CONFIG_PATH] [-d SIM_DATA_PATHS [SIM_DATA_PATHS ...]]
                 [-o OUTPUT_DIRECTORY] [--overwrite] [--no-cleanup]
                 [--keep-split] [--parallel] [-v] [--no-progress]
                 [--ndirs NDIRS] [--nphases NPHASES] [--nordersout NORDERSOUT]
                 [--angle0 ANGLE0] [--ls LS] [--na NA] [--nimm NIMM]
                 [--wiener WIENER] [--zoomfact ZOOMFACT] [--zzoom ZZOOM]
                 [--background BACKGROUND] [--usecorr USECORR]
                 [--k0angles K0ANGLES K0ANGLES K0ANGLES] [--otfRA OTFRA]
                 [--otfPerAngle OTFPERANGLE] [--fastSI FASTSI]
                 [--k0searchAll K0SEARCHALL] [--norescale NORESCALE]
                 [--equalizez EQUALIZEZ] [--equalizet EQUALIZET]
                 [--dampenOrder0 DAMPENORDER0] [--nosuppress NOSUPPRESS]
                 [--nokz0 NOKZ0] [--gammaApo GAMMAAPO]
                 [--explodefact EXPLODEFACT] [--nofilterovlps NOFILTEROVLPS]
                 [--deskew DESKEW] [--deskewshift DESKEWSHIFT]
                 [--noRecon NORECON] [--cropXY CROPXY] [--xyres XYRES]
                 [--zres ZRES] [--zresPSF ZRESPSF] [--wavelength WAVELENGTH]
                 [-h]

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
  --no-progress         turn off progress bars (only has an affect if tqdm is
                        installed)
  -h, --help            show this help message and exit

Overrides:
  Arguments that override configured values. Defaults stated are only used
  if no value is given or configured.

  --ndirs NDIRS         number of SIM directions
  --nphases NPHASES     number of pattern phases per SIM direction
  --nordersout NORDERSOUT
                        number of output SIM orders (must be <= nphases//2;
                        safe to ignore usually)
  --angle0 ANGLE0       angle of the first SIM angle in radians
  --ls LS               line spacing of SIM pattern in microns
  --na NA               detection objective's numerical aperture
  --nimm NIMM           refractive index of immersion medium
  --wiener WIENER       Wiener constant; lower value leads to higher
                        resolution and noise;
  --zoomfact ZOOMFACT   lateral zoom factor in the output over the input
                        images;
  --zzoom ZZOOM         axial zoom factor; almost never needed
  --background BACKGROUND
                        camera readout background
  --usecorr USECORR     use a flat-field correction file if provided
  --k0angles K0ANGLES K0ANGLES K0ANGLES
                        user these pattern vector k0 angles for all
  --otfRA OTFRA         using rotationally averaged OTF; otherwise using
  --otfPerAngle OTFPERANGLE
                        using one OTF per SIM angle; otherwise one OTF is
  --fastSI FASTSI       SIM image is organized in Z->Angle->Phase order;
  --k0searchAll K0SEARCHALL
                        search for k0 at all time points
  --norescale NORESCALE
                        no bleach correction
  --equalizez EQUALIZEZ
                        bleach correction for z
  --equalizet EQUALIZET
                        bleach correction for time
  --dampenOrder0 DAMPENORDER0
                        dampen order-0 in final assembly; do not use for 2D
                        SIM; good choice for high-background images
  --nosuppress NOSUPPRESS
                        do not suppress DC singularity in the result (good
                        choice for 2D/TIRF data)
  --nokz0 NOKZ0         not using kz=0 plane of the 0th order in the final
                        assembly (mostly for debug)
  --gammaApo GAMMAAPO   output apodization gamma; 1.0 means triangular
  --explodefact EXPLODEFACT
                        artificially exploding the reciprocal-space
  --nofilterovlps NOFILTEROVLPS
  --deskew DESKEW       Deskew angle; if not 0.0 then perform deskewing before
                        processing
  --deskewshift DESKEWSHIFT
                        If deskewed, the output image's extra shift in X
                        (positive->left)
  --noRecon NORECON     No reconstruction will be performed; useful when
                        combined with "deskew":
  --cropXY CROPXY       Crop the X-Y dimension to this number; 0 means no
                        cropping
  --xyres XYRES         x-y pixel size (only used for TIFF files)
  --zres ZRES           z pixel size (only used for TIFF files)
  --zresPSF ZRESPSF     z pixel size (used in PSF TIFF files)
  --wavelength WAVELENGTH
                        emission wavelength in nanometers (only used for TIFF
                        files)
```

##### dv2tiff
Accepts a list of DV files to be converted to TIFFs (with some OME metadata).

##### otf-view
Accepts a list of OTF files (DV or TIFF), the amplitude and phase of which will be plotted.
