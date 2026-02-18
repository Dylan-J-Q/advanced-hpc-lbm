#!/bin/bash
#SBATCH --job-name=this-is-a-job-name
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=144
#SBATCH --exclusive 
#SBATCH --ntasks-per-node=1



make
./d2q9-bgk input_128x128.params obstacles_128x128.dat
./d2q9-bgk input_128x256.params obstacles_128x256.dat
./d2q9-bgk input_256x256.params obstacles_256x256.dat
./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat