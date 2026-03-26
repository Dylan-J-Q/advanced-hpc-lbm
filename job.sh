#!/bin/bash
#SBATCH --job-name=d2q9-bgk-mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:30
#SBATCH --exclusive

module load PrgEnv-gnu

make clean
make

srun ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat