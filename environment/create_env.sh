#!/bin/bash

set -e

echo ">>> Starting Conda environment creation from environment.yaml..."

conda update -n base -c defaults conda -y
conda env create -f $PWD/environment/environment.yaml

echo ">>> Environment 'neuro' created successfully!"