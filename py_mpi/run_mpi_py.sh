#!/bin/bash
#SBATCH --job-name=mpi_bicgstab_py
#SBATCH --ntasks=112
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00

module load mpi
module load python

OUTFILE="mpi_py_results_$(date +%Y%m%d_%H%M%S).log"

echo "Writing results to $OUTFILE"
echo "===== job started at $(date) =====" >> "$OUTFILE"

for np in 1 2 4 8 12 16 24 32 48 64 112; do
    echo "===== np = $np =====" | tee -a "$OUTFILE"
    mpirun -np $np python bicgstab_mpi.py >>"$OUTFILE" 2>>"$OUTFILE"
done

echo "===== job finished at $(date) =====" >> "$OUTFILE"
