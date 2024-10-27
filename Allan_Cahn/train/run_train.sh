#!/bin/bash

python3 G_convergence_lz.py --seed=1
python3 G_convergence_lz.py --seed=2
python3 G_convergence_lz.py --seed=3

python3 G_convergence_lx.py --seed=1
python3 G_convergence_lx.py --seed=2
python3 G_convergence_lx.py --seed=3


