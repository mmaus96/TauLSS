#!/bin/bash
#
# Set up a conda environment for Cobaya.
# This version is customized for NERSC.  Cloning the
# nersc-mpi4py environment ensures mpi4py is properly
# included in its NERSC form.
#
module load python
#
# conda deactivate
# conda remove --name cobaya --all
# conda clean --all
#
conda create --name cobaya_tau --clone nersc-mpi4py
#
# Switch to the environment.
conda activate cobaya_tau
#
# Install some basic stuff
conda install -c conda-forge numpy scipy matplotlib -y
conda install -c conda-forge astropy sympy pandas cython -y
conda install -c conda-forge pyfftw -y
conda install -c conda-forge healpy -y
#
# Set up the environment for Jupyter.
conda install -c conda-forge ipykernel ipython jupyter -y
python3 -Xfrozen_models=off -m ipykernel install --user \
        --name cobaya_tau --display-name cobaya_tau

python3 -m pip install act_dr6_lenslike
#
# Now install Cobaya
python3 -m pip install cobaya --upgrade
#python3 -m pip install git+https://github.com/CobayaSampler/cobaya


#
# and any "cosmo" packages it wants
rm -rf  $SCRATCH/Cobaya/Packages
cobaya-install cosmo -p $SCRATCH/Cobaya/Packages
# you may need to "upgrade" the packages:
cobaya-install cosmo --upgrade -p $SCRATCH/Cobaya/Packages
#
# If you want to use ACT likelihoods, first pip install:
# python3 -m pip install act_dr6_lenslike
# then you can use cobaya-install.
# Easiest is from a YAML file containing:
# likelihood:
#    act_dr6_lenslike.ACTDR6LensLike:
#        lens_only: False
#        stop_at_error: True
#        lmax: 4000
#        variant: act_baseline
# You also need to install the actual data that the likelihood
# uses which can be done with the "get_act_data.sh" script.
#
# Install velocileptors. 
python3 -m pip install velocileptors # --upgrade
# and findiff for the Taylor series emulators
python3 -m pip install findiff # --upgrade
#

#Optional:
python -m pip install git+https://github.com/cosmodesi/cosmoprimo
pip install -U jax
pip install interpax
