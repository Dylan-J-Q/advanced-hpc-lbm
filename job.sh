#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=1

make
./d2q9-bgk input_128x128.params obstacles_128x128.dat


