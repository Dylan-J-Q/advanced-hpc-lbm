#!/bin/bash
#SBATCH --job-name=d2q9-bgk-scaling
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=scaling_%j.out

module load PrgEnv-gnu

make clean
make

# ── Problem sizes ──
INPUTS=(
  "input_1024x1024.params obstacles_1024x1024.dat"
  "$PROJECTDIR/public/d2q9-bgk-inputs/input_2048x2048.params $PROJECTDIR/public/d2q9-bgk-inputs/obstacles_2048x2048.dat"
  "$PROJECTDIR/public/d2q9-bgk-inputs/input_4096x4096.params $PROJECTDIR/public/d2q9-bgk-inputs/obstacles_4096x4096.dat"
)

LABELS=("1024x1024" "2048x2048" "4096x4096")

# ── Rank counts for strong scaling ──
RANKS=(1 4 16 36 72 144 288 576)

echo "========================================"
echo "  Strong Scaling Study"
echo "  Date: $(date)"
echo "  Nodes: $SLURM_JOB_NUM_NODES"
echo "========================================"
echo ""

for i in "${!INPUTS[@]}"; do
  PARAMFILE=$(echo "${INPUTS[$i]}" | cut -d' ' -f1)
  OBSTFILE=$(echo "${INPUTS[$i]}" | cut -d' ' -f2)
  LABEL="${LABELS[$i]}"

  echo "----------------------------------------"
  echo "  Problem size: $LABEL"
  echo "----------------------------------------"

  for NP in "${RANKS[@]}"; do
    echo ""
    echo ">> $LABEL | ranks=$NP"
    srun --ntasks=$NP ./d2q9-bgk "$PARAMFILE" "$OBSTFILE"
    echo ""
  done
done
