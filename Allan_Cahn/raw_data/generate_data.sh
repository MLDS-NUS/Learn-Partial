#!/bin/bash
python3 batch_solver_RK4.py  --save_path='data_train.pt' --bs=400 
python3 batch_solver_RK4.py  --save_path='data_test.pt' --bs=20 
python3 batch_solver_RK4.py  --save_path='data_test_interpolate.pt' --bs=50 
python3 batch_solver_RK4.py  --save_path='data_plot.pt' --bs=36 

