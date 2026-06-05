#!/bin/bash
#
if [ ! -d data ]; then
    echo "data not found.  Creating it."
    mkdir ./data
fi
if [ ! -d data/DR1 ]; then
    echo "data/DR1 not found.  Creating it."
    mkdir ./data/DR1
fi
cp -r /global/cfs/cdirs/desi/users/mmaus/C3PO/data/spec_z_dat ./data/DR1/