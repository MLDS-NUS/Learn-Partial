# !/bin/bash
python3 G_process_lx.py --cuda_device=2 --d=20 &
python3 G_process_lx.py --cuda_device=2 --d=25 &
python3 G_process_lx.py --cuda_device=2 --d=50 &
python3 G_process_lx.py --cuda_device=2 --d=75 &
python3 G_process_lx.py --cuda_device=2 --d=100 &
wait 

python3 G_process_lz.py --cuda_device=2 --n_list=600 --n_list=1500 --n_list=3000 --seed_list=1 --seed_list=2 --seed_list=3 &
python3 G_process_lz.py --cuda_device=2 --n_list=6000 --n_list=12000 --n_list=24000 --seed_list=1 --seed_list=2 --seed_list=3 &
wait
