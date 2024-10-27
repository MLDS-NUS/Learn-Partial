#!/bin/bash

# download the data for GFINNs model 
git clone https://github.com/zzhang222/gfinn_gc.git


# Data Generation 
cd raw_data
bash generate_data.sh

# Process raw data
cd ..
python3 utils/raw_lammps_data_process.py

# train the autoencoder 
cd train
# python3 AE.py
python3 AE_process_batch.py


# Train the macroscopic dynamcis model 
python3 G_lz.py
python3 G_lx.py

# bash run_train.sh

# test 
cd ../results_process
python3 results.py
