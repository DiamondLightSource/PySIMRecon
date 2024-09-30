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
  - At the time of writing, the latest version of cudasirecon (1.2.0) requires CUDA 11.8+. Older versions of cudasirecon have a lower requirement of CUDA 10.2+ but are unsupported by this package.
  - The CUDA version that cudasirecon is built for is dependent on conda-forge's support.
  - It is recommended to use the latest GPU drivers available for you system. Please see [cudasirecon's README](https://github.com/scopetools/cudasirecon/blob/main/README.md#gpu-requirements) for more details about CUDA and driver versions.
- Unfortunately, macOS is not supported.

##### Steps

This is not yet published on conda-forge, so the installation process is fairly manual.

1. Within a conda terminal, create a new environment `conda create -n pysimrecon_env python=3.12` and activate it `conda activate pysimrecon_env`. The environment has been called `pysimrecon_env` in this example but this is not a requirement.
2. Clone (or download) this repository.
3. Navigate to where you've cloned the repo within your terminal.
4. Install the requirements from the conda_requirements.txt file `conda install -c conda-forge --file conda_requirements.txt`. If there are any issues with this step, please refer to the [requirements](#requirements) section.
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

```

##### sim-recon

```

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
