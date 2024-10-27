# python3 AE.py --N_atoms=800 --input_dim=4800 --lr=1e-4
# python3 AE.py --N_atoms=6400 --input_dim=38400 --lr=1e-3 --cuda_device=1 --train_bs=32 
# python3 AE.py --N_atoms=21600 --input_dim=129600 --lr=1e-3 --lbd=1e-3 --cuda_device=1 --train_bs=32 
# python3 AE.py --N_atoms=51200 --input_dim=307200 --lr=1e-3 --lbd=1e-3 --cuda_device=1 --train_bs=32  
# only 1 layer 

#!/bin/bash


# data_size_list=(1250 1500 1750)
# MAX_JOBS=6
# current_jobs=0
# for N in "${data_size_list[@]}"; do
#     python3 G_lz.py --cuda_device=0 --train_size="$N" &
#     ((current_jobs++))
#     if (( current_jobs >= MAX_JOBS )); then
#         wait
#         current_jobs=0
#     fi
# done
# wait


data_size_list=(500 750 1000 1500)
MAX_JOBS=6
current_jobs=0
for N in "${data_size_list[@]}"; do
    python3 G_lz.py --cuda_device=0 --N_atoms=2700 --AE_folder='../checkpoints/AE_dim_32_atoms_2700_2024_05_20_23:16:28' --train_size="$N" &
    ((current_jobs++))
    if (( current_jobs >= MAX_JOBS )); then
        wait
        current_jobs=0
    fi
done
wait


# data_size_list=(500 750 1000 1500)
# MAX_JOBS=6
# current_jobs=0
# for N in "${data_size_list[@]}"; do
#     python3 G_lz.py --cuda_device=0 --N_atoms=6400 --AE_folder='./checkpoints/AE_dim_32_atoms_6400_2024_05_20_22:33:03' --train_size="$N" &
#     ((current_jobs++))
#     if (( current_jobs >= MAX_JOBS )); then
#         wait
#         current_jobs=0
#     fi
# done
# wait


# data_size_list=(250 500)
# MAX_JOBS=6
# current_jobs=0
# for N in "${data_size_list[@]}"; do
#     python3 G_lz.py --cuda_device=0 --N_atoms=21600 --AE_folder='./checkpoints/AE_dim_32_atoms_21600_2024_05_19_08:18:37' --train_size="$N" &
#     ((current_jobs++))
#     if (( current_jobs >= MAX_JOBS )); then
#         wait
#         current_jobs=0
#     fi
# done
# wait
