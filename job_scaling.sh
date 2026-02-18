#!/bin/bash
#SBATCH --job-name=d2q9-scaling
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1

make clean
make

export OMP_PROC_BIND=true
export OMP_PLACES=cores

THREADS=(1 2 4 8 16 32 64 128 144)

for t in "${THREADS[@]}"
do
    echo "======================================="
    echo "Running with $t threads"
    echo "======================================="

    export OMP_NUM_THREADS=$t

    ./d2q9-bgk input_256x256.params obstacles_256x256.dat
done
