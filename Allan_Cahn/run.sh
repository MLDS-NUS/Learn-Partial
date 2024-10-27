#!/bin/bash

# Data Generation 
cd raw_data
bash generate_data.sh

# train the autoencoder 
cd ../train
# python3 AE.py
python3 AE_process_batch.py


# Train the macroscopic dynamcis model 
# python3 G_convergence_lz.py
# python3 G_convergence_lx.py

bash run_train.sh

# test 
cd ../results_process
bash run_process.sh
