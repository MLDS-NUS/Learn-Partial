# Experiments on Lennard-Jones System  

This is an implementation of our method on the Lennard-Jones System. 

## User Manual
To run all the Python scripts for the Lennard-Jones experiment together, execute the `run.sh` file:
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
python3 G_lz.py
```

Train the macorsocpic dynamcis model with loss $\mathcal{L}_{\mathbf{x}}$ by running:
```bash
cd train
python3 G_lx.py
```

### Evaluation 
To evaluate the performace of the models trained with loss $\mathcal{L}_{\mathbf{z}}$ and $\mathcal{L}_{\mathbf{x}}$, 
follow these steps:
```bash
cd ../results_process
python3 results.py
```
To reproduce the figures relating of Lennard-Jones experiment in the paper, run `plot.ipynb`. 


## Benchmark model
We have provided the model that are used for the data reported in the paper. All the models are located in the `checkpoints` folder.