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
