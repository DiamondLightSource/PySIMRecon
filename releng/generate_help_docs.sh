#! /bin/bash

REPO_DIR="$(realpath -- "$(dirname -- "$(dirname -- "${BASH_SOURCE[0]}")")")"
cd "$REPO_DIR" || exit 1
mkdir -p "docs/help"
export COLUMNS=80 # Sets line wrap for Python argparse
python -m sim_recon.cli.parsing.otf '--help' >docs/help/sim_otf.txt
python -m sim_recon.cli.parsing.recon '--help' >docs/help/sim_recon.txt
python -m sim_recon.cli.parsing.otf_view '--help' >docs/help/otf_view.txt
