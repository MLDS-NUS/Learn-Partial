#!/bin/bash

N_list=(100)
seed_list=(1 2 3)

d_list=(20)
n_list=(3000 7500 15000 30000 60000 120000)

# d_list=(25)
# n_list=(2400 600 12000 24000 48000 96000)

# d_list=(50)
# n_list=(1200 3000 6000 12000 24000 48000)

# d_list=(75)
# n_list=(800 2000 4000 8000 16000 32000)

# d_list=(100)
# n_list=(600 1500 3000 6000 12000 24000)



# Maximum number of parallel jobs
MAX_JOBS=6
current_jobs=0

for N in "${N_list[@]}"; do
    for n in "${n_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            for d in "${d_list[@]}"; do
                # Run the python script 
                python3 G_convergence_lx.py --cuda_device=0 --N="$N" --n="$n" --seed="$seed" --d="$d" &

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
done

# Wait for any remaining jobs to complete
wait
