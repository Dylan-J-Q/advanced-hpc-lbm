#!/bin/bash
#SBATCH --job-name=d2q9-bgk-mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --exclusive

module load PrgEnv-gnu

make clean
make

srun ./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
make check
srun ./d2q9-bgk "$PROJECTDIR/public/d2q9-bgk-inputs/input_2048x2048.params" "$PROJECTDIR/public/d2q9-bgk-inputs/obstacles_2048x2048.dat"
srun ./d2q9-bgk "$PROJECTDIR/public/d2q9-bgk-inputs/input_4096x4096.params" "$PROJECTDIR/public/d2q9-bgk-inputs/obstacles_4096x4096.dat"
