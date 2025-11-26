#!/bin/bash
#SBATCH --job-name=mpi_bicgstab
#SBATCH --ntasks=112
#SBATCH --time=00:30:00

OUTFILE="mpi_results_$(date +%Y%m%d_%H%M%S).log"

echo "===== job started at $(date) =====" >> "$OUTFILE"

module load mpi

for np in 1 2 4 8 12 16 24 32 48 64 112; do
    /usr/bin/time -f "np=$np real=%e user=%U sys=%S" \
        mpirun -np $np ./bicgstab_mpi \
        1>/dev/null 2>tmp_time.err

    grep '^np=' tmp_time.err >>"$OUTFILE"
    rm tmp_time.err
done

echo "===== job finished at $(date) =====" >> "$OUTFILE"
