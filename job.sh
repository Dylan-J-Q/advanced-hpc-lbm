#!/bin/bash
#SBATCH --job-name=this-is-a-job-name
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=144
#SBATCH --exclusive 
#SBATCH --ntasks-per-node=1

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=144
export OMP_DYNAMIC=false
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

make clean
make
./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat