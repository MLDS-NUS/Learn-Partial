#!/bin/bash

# Data Generation 
cd raw_data
bash generate_data.sh

# Process raw data
cd ..
python3 utils/raw_data_preprocess.py

# train the autoencoder 
cd train
# python3 AE.py
python3 AE_process_batch.py


# Train the macroscopic dynamcis model 

# python3 G_convergence_lz.py
# python3 G_convergence_lx.py

bash run_train_lz.sh
bash run_train_lx.sh

# test 
cd ../results_process
bash run_G_process.sh
