#!/bin/bash
#SBATCH --job-name=d2q9-bgk-roofline
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --output=d2q9-bgk-roofline.out

set -euo pipefail

make clean
make

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=144
export OMP_DYNAMIC=false
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

PARAMS="input_1024x1024.params"
OBS="obstacles_1024x1024.dat"

OUTDIR="perf_out"
mkdir -p "${OUTDIR}"

CORE_EVENTS="duration_time,cycles,instructions,r80C0,r80C1"
SCF_EVENTS="{nvidia_scf_pmu_0/cmem_wr_total_bytes/,nvidia_scf_pmu_0/cmem_rd_data/},{nvidia_scf_pmu_1/cmem_wr_total_bytes/,nvidia_scf_pmu_1/cmem_rd_data/}"

PERF_STAT_CSV="${OUTDIR}/perf_stat_roofline.csv"
PERF_DATA="${OUTDIR}/perf.data"

perf stat -a -x, \
  -e ${CORE_EVENTS},${SCF_EVENTS} \
  -o "${PERF_STAT_CSV}" \
  -- \
  perf record -F 999 -g --call-graph dwarf -o "${PERF_DATA}" -- \
    ./d2q9-bgk "${PARAMS}" "${OBS}"

perf report --stdio -i "${PERF_DATA}" --no-children > "${OUTDIR}/perf_report.txt"
perf annotate -i "${PERF_DATA}" > "${OUTDIR}/perf_annotate.txt"

SUMMARY_TXT="${OUTDIR}/roofline_summary.txt"

awk -F, '
function trim(s){ gsub(/^[ \t]+|[ \t]+$/, "", s); return s }

{
  val=trim($1); unit=trim($2); evt=trim($3);

  # duration_time is in ns in NVIDIA examples; treat as ns -> seconds
  if (evt=="duration_time") duration_ns=val+0;

  if (evt=="r80C0") fp_scale=val+0;
  if (evt=="r80C1") fp_fixed=val+0;

  if (evt ~ /nvidia_scf_pmu_0\/cmem_wr_total_bytes\//) wr_bytes += val+0;
  if (evt ~ /nvidia_scf_pmu_1\/cmem_wr_total_bytes\//) wr_bytes += val+0;

  if (evt ~ /nvidia_scf_pmu_0\/cmem_rd_data\//) rd_beats += val+0;
  if (evt ~ /nvidia_scf_pmu_1\/cmem_rd_data\//) rd_beats += val+0;
}
END{
  if (duration_ns<=0) duration_ns=1;

  time_s = duration_ns / 1e9;

  flops = fp_scale + fp_fixed;

  # each data beat transfers up to 32 bytes on Grace SCF PMU
  rd_bytes = rd_beats * 32.0;

  bytes = wr_bytes + rd_bytes;

  ai = (bytes>0)? (flops/bytes) : 0;

  gflops = flops / time_s / 1e9;
  gbytes = bytes / time_s / 1e9;

  print "=== Roofline inputs (Grace / perf) ===";
  print "time_s:           " time_s;
  print "FP_SCALE_OPS:     " fp_scale "  (r80C0)";
  print "FP_FIXED_OPS:     " fp_fixed "  (r80C1)";
  print "FLOPs_total:      " flops;
  print "WR_bytes_total:   " wr_bytes;
  print "RD_beats_total:   " rd_beats;
  print "RD_bytes_est:     " rd_bytes "  (beats*32)";
  print "Bytes_total_est:  " bytes;
  print "ArithmeticIntensity(F/B): " ai;
  print "GFLOP/s:          " gflops;
  print "GB/s:             " gbytes;
}
' "${PERF_STAT_CSV}" | tee "${SUMMARY_TXT}"

echo "Done. Roofline CSV: ${PERF_STAT_CSV}"
echo "Roofline summary:   ${SUMMARY_TXT}"
echo "Hotspots:           ${OUTDIR}/perf_report.txt"
