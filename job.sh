#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1


make

./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
./d2q9-bgk "$PROJECTDIR/public/d2q9-bgk-inputs/input_2048x2048.params" "$PROJECTDIR/public/d2q9-bgk-inputs/obstacles_2048x2048.dat"