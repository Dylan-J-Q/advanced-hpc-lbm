#!/bin/bash
#SBATCH --job-name=d2q9-bgk-scaling
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=scaling_%j.out

module load PrgEnv-gnu

make clean
make

CSV="scaling_results.csv"
echo "problem_size,ranks,reynolds,init_time,compute_time,collate_time,total_time" > "$CSV"

INPUTS=(
  "input_1024x1024.params obstacles_1024x1024.dat"
  "$PROJECTDIR/public/d2q9-bgk-inputs/input_2048x2048.params $PROJECTDIR/public/d2q9-bgk-inputs/obstacles_2048x2048.dat"
  "$PROJECTDIR/public/d2q9-bgk-inputs/input_4096x4096.params $PROJECTDIR/public/d2q9-bgk-inputs/obstacles_4096x4096.dat"
)

LABELS=("1024x1024" "2048x2048" "4096x4096")

RANKS=(1 16 32 64 128 256 512 576)

echo "========================================"
echo "  Strong Scaling Study"
echo "  Date: $(date)"
echo "========================================"

for i in "${!INPUTS[@]}"; do
  PARAMFILE=$(echo "${INPUTS[$i]}" | cut -d' ' -f1)
  OBSTFILE=$(echo "${INPUTS[$i]}" | cut -d' ' -f2)
  LABEL="${LABELS[$i]}"

  echo ""
  echo "--- Problem: $LABEL ---"

  for NP in "${RANKS[@]}"; do
    echo ">> $LABEL | ranks=$NP"

    OUTPUT=$(srun --ntasks=$NP ./d2q9-bgk "$PARAMFILE" "$OBSTFILE" 2>&1)
    echo "$OUTPUT"

    # parse timing values from the program output
    REYNOLDS=$(echo "$OUTPUT" | grep "Reynolds"      | awk '{print $NF}')
    INIT=$(echo "$OUTPUT"     | grep "Init time"     | awk '{print $NF}')
    COMPUTE=$(echo "$OUTPUT"  | grep "Compute time"  | awk '{print $NF}')
    COLLATE=$(echo "$OUTPUT"  | grep "Collate time"  | awk '{print $NF}')
    TOTAL=$(echo "$OUTPUT"    | grep "Total time"    | awk '{print $NF}')

    echo "$LABEL,$NP,$REYNOLDS,$INIT,$COMPUTE,$COLLATE,$TOTAL" >> "$CSV"
    echo ""
  done
done

echo "========================================"
echo "  Complete: $(date)"
echo "  Results in: $CSV"
echo "========================================"