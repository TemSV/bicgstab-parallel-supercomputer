#!/bin/bash
#SBATCH --job-name=bicgstab_omp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --partition=tornado

module load compiler/gcc/11

OUTFILE="omp_results_$(date +%Y%m%d_%H%M%S).log"
echo "===== job started at $(date) =====" >> "$OUTFILE"

for t in 1 2 4 8 12 16 24 32 48; do
    echo "===== threads = $t =====" | tee -a "$OUTFILE"
    OMP_NUM_THREADS=$t ./bicgstab_omp >>"$OUTFILE" 2>>"$OUTFILE"
done

echo "===== job finished at $(date) =====" >> "$OUTFILE"
