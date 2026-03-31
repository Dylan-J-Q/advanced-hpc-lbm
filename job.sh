#!/bin/bash
#SBATCH --job-name=d2q9-bgk-mpi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

module load PrgEnv-gnu

make clean
make

srun ./d2q9-bgk "$PROJECTDIR/public/d2q9-bgk-inputs/input_2048x2048.params" "$PROJECTDIR/public/d2q9-bgk-inputs/obstacles_2048x2048.dat"
srun ./d2q9-bgk "$PROJECTDIR/public/d2q9-bgk-inputs/input_4096x4096.params" "$PROJECTDIR/public/d2q9-bgk-inputs/obstacles_4096x4096.dat"
