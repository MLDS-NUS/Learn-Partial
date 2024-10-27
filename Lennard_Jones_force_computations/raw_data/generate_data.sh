#!/bin/bash

SEED=$1
if [ -z "$SEED" ]; then
    SEED=1  # Set the seed to 1 if no seed is provided
fi

RANDOM=$SEED
num_runs=50

lammps_exe="lmp" 
lammps_template="lj.lammps"

# Loop through runs
for ((i=1; i<=$num_runs; i++))
do
    Trandom_seed=$RANDOM
    random_float=$(echo "scale=4; $Trandom_seed / 32767" | bc)  # Use bc for floating-point precision
    random_T=$(echo "scale=4; $random_float * 1 + 0.5" | bc)
    output_xyz="output_run_${i}.xyz"

    cp "$lammps_template" "lammps_input_run_${i}.in"

    sed -i "s/variable outputfile string.*/variable outputfile string \"$output_xyz\"/" "lammps_input_run_${i}.in"
    sed -i "s/variable T equal.*/variable T equal \"$random_T\"/" "lammps_input_run_${i}.in"
    
    $lammps_exe -in "lammps_input_run_${i}.in"

    echo "Run $i completed with random seed $random_seed. Output: $output_xyz"
done
