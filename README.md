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
