### Installation

This package requires Conda for package management. This project recommends using [miniforge](https://conda-forge.org/download/) from conda-forge.

This is not yet published on conda-forge, so the installation process is fairly manual.

1. Within a conda terminal, create a new environment `conda create -n pysimrecon_env python=3.12` and activate it `conda activate pysimrecon_env`. The environment has been called `pysimrecon_env` in this example but this is not a requirement.
2. Clone (or download) this repository.
3. Navigate to where you've cloned the repo within your terminal.
4. Install the requirements from the conda_requirements.txt file `conda install --file conda_requirements.txt`
5. Install the package with pip, now that the requirements have been met `python -m pip install .` The argument `-e` can be added if you want it install it in editable mode.
