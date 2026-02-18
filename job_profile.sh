#!/bin/bash
#SBATCH --job-name=d2q9-perf
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1

set -euo pipefail

make clean
make

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=144

PARAMS="input_1024x1024.params"
OBS="obstacles_1024x1024.dat"

OUTDIR="perf_out"
mkdir -p "${OUTDIR}"

perf record -F 999 -g --call-graph dwarf -o "${OUTDIR}/perf.data" -- \
  ./d2q9-bgk "${PARAMS}" "${OBS}"

perf report --stdio -i "${OUTDIR}/perf.data" --no-children > "${OUTDIR}/perf_report.txt"

perf annotate -i "${OUTDIR}/perf.data" > "${OUTDIR}/perf_annotate.txt"

echo "Done. Reports in ${OUTDIR}/"
