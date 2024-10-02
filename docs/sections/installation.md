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
