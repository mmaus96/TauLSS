#!/bin/bash
#
# Get the data that the ACT DR6 likelihood depends upon.
# These data should be installed where Cobaya expects to
# find them.  If the Cobaya packages were installed with:
# cobaya-install cosmo -p $SCRATCH/Cobaya/Packages
# then these data go in
# $SCRATCH/Cobaya/act_dr6_lenslike/data/
# This is assumed below.
#
# Web location of the data
url1=https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data
url2=ACT_dr6_likelihood_v1.2.tgz
# Fetch the data and unpack it.
cd $SCRATCH/Cobaya
mkdir -p act_dr6_lenslike/data
cd $SCRATCH/Cobaya/act_dr6_lenslike/data
wget ${url1}/${url2}
tar -zxvf ${url2}
rm ${url2}
#
# Now move it to where Cobaya expects to find it.
#
cd $SCRATCH/Cobaya/act_dr6_lenslike/data/
rsync -av v1.2 $SCRATCH/Cobaya/Packages/data/ACT_dr6_likelihood/
#
