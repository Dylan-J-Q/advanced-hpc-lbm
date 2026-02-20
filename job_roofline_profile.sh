#!/bin/bash
#SBATCH --job-name=grace-carm-roofline
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --output=grace-carm-roofline.out

# Get the carm-roofline repository
git clone https://github.com/addy419/carm-roofline
make clean
make

WORK_DIR="$(pwd)"

cd carm-roofline
# Load nvhpc compiler
module load nvidia
# Run the roofline
# Example:
# threads - 72
# precison - double
# loads only
# set sufficient l3 and dram kbytes (1 socket)
OMP_NUM_THREADS=72 python3 run.py ./config/isambard-grace.conf \
  --isa sve --threads 144 --precision dp -tl1 1 -tl2 1 \
  --only_ld --l3_kbytes 116736 --dram_kbytes 34504704 \
  --test roofline

source /projects/b35cg/carm.sh
OMP_NUM_THREADS=72 python3 DBI_AI_Calculator.py \
  --roi --name isambard-grace --threads 144 \
  /projects/b35cg/dynamorio \
  "$WORK_DIR/d2q9-bgk" "$WORK_DIR/input_1024x1024.params" "$WORK_DIR/obstacles_1024x1024.dat"