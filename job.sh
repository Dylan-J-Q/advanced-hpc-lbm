#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --exclusive 
#SBATCH --ntasks-per-node=1


make

./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
