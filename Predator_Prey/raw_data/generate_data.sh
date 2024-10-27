#!/bin/bash

python3 batch_solver.py  --save_path='data_train.pt' --maxit=3000 --bs=1600 --dt=0.01 
python3 batch_solver.py  --save_path='data_test.pt' --maxit=3000 --bs=20 --dt=0.01 
python3 batch_solver_RK4.py  --save_path='data_test_interpolate.pt' --maxit=3000 --dt=0.01 --bs=100
python3 batch_solver_RK4.py  --save_path='plot.pt' --maxit=3000 --dt=0.01 --bs=100 
