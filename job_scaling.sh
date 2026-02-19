#!/bin/bash
#SBATCH --job-name=d2q9-scaling
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1

make clean
make

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=144
export OMP_DYNAMIC=false
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

THREADS=(1 2 4 8 16 32 64 128 144)

for t in "${THREADS[@]}"
do
    echo "======================================="
    echo "Running with $t threads"
    echo "======================================="

    export OMP_NUM_THREADS=$t

    ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
done
