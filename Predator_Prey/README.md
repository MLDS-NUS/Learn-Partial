# Experiments on Predator-Prey System  

This is an implementation of our method on the Predator-Prey System. 

## User Manual
To run all the Python scripts for the Predator-Prey experiment together, execute the `run.sh` file:
```bash
bash run.sh
```
If you prefer to perform the experiment step by step, follow these instructions:

### Generate data
To generate the data for training and testing, follow these steps: 
```bash
cd raw_data 
bash generate_data.sh
cd ..
python3 utils/raw_data_preprocess.py
```

### Train an autoencoder
Train an autoencoder to find the closure of the macroscopic observables by runnning: 
```bash
cd train
python3 AE.py
python3 AE_process_batch.py
```

### Train macorsocpic dynamcis model
Train the macorsocpic dynamcis model with loss $\mathcal{L}_{\mathbf{z}}$ by running:
```bash
cd train
python3 G_convergence_lz.py
```

Train the macorsocpic dynamcis model with loss $\mathcal{L}_{\mathbf{x}}$ by running:
```bash
cd train
python3 G_convergence_lx.py
```

### Evaluation 
To evaluate the performace of the models trained with loss $\mathcal{L}_{\mathbf{z}}$ and $\mathcal{L}_{\mathbf{x}}$, 
follow these steps:
```bash
cd ../results_process
bash run_G_process.sh
```
To reproduce the figures relating of Predator-Prey experiment in the paper, run `plot.ipynb`. 


## Benchmark model
We have provided the model that are used for the data reported in the paper. 
The autoencoder model is located in the `checkpoints` folder, 
and the macriscipic dynamics models are located in the `checkpoints_100_lx_saved` folder.  