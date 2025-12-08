#!/bin/bash
#SBATCH --job-name=pthreads_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00

OUTFILE="pthreads_results_$(date +%Y%m%d_%H%M%S).log"

echo "===== job started at $(date) =====" >> "$OUTFILE"

for nt in 1 2 4 8 12 16 24 32 48; do
    export OMP_NUM_THREADS=$nt

    /usr/bin/time -f "threads=$nt real=%e user=%U sys=%S" \
        ./pthreads \
        1>/dev/null 2>tmp_time.err

    grep '^threads=' tmp_time.err >>"$OUTFILE"
    rm tmp_time.err
done

echo "===== job finished at $(date) =====" >> "$OUTFILE"
