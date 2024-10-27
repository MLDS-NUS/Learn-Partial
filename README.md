# Code for *Learning Macroscopic Dynamics from Partial Microscopic Observations*


This repository contains the implementation of the methods and experiments from the paper 'Learning Macroscopic Dynamics from Partial Microscopic Observations' .

## Description
This repository contains four experiments, each organized into a separate folder. We introduce the code and scripts needed to reproduce the experiments in detail in each respective folder. 

* `Predator_Prey`: The implementation of the Predator-Prey experiment in the paper. 

* `Lennard_Jones`: The implementation of the Lennard-Jones experiment in the paper. We consider Lennard-Jones system with 800 and 51200 atoms. 

* `Allen-Cahn`: The implementation of the Allen-Cahn experiment in the paper. 

* `Lennard_Jones_force_computations`: We reproduce Figure 4 of the paper in this folder. 
Lennard-Jones system with 800, 2700, 6400, 21600 atoms are considered. 


## Getting Started
To get started, clone this repository to your local machine using the following command:
```bash
https://github.com/MLDS-NUS/Learn-from-Partial-Microscopic-dynamics.git
```

### Installing
To initialize the environment, create a new environment and install the prerequisite dependencies:
```bash
conda create -n newenv python=3.9.18
conda activate newenv
pip install -r requirements.txt
```
### Dependencies
* LAMMPS: We use LAMMPS (stable_2Aug2023) for data generation of Lennard-Jones experiment.
* CUDA Version: 12.4
* GPU Model: NVIDIA GeForce RTX 3090


## Executing program
Detailed step-by-step instructions are provided within each folder

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).


## References

**to be done**
