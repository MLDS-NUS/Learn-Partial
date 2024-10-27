#!/bin/bash

N_list=(100)
n_list=(600 1500 3000 6000 12000 24000)
seed_list=(1 2 3)

# Maximum number of parallel jobs
MAX_JOBS=6
current_jobs=0

for N in "${N_list[@]}"; do
    for n in "${n_list[@]}"; do
        for seed in "${seed_list[@]}"; do
        
            # Run the python script
            python3 G_convergence_lz.py --cuda_device=0 --N="$N" --n="$n" --seed="$seed" &

            # Increment the jobs counter
            ((current_jobs++))

            # If we've reached the maximum number of jobs, wait for all to complete
            if (( current_jobs >= MAX_JOBS )); then
                wait
                current_jobs=0
            fi
        done
    done
done

# Wait for any remaining jobs to complete
wait
