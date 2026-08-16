#!/usr/bin/env bash
# Usage: ./run_thermal.sh [threads]   (default: all physical cores)
set -euo pipefail

# threads="${1:-$(nproc)}"
threads="3"
# SPRAL (Ipopt's parallel linear solver) reads these at library load time,
# so they must be exported BEFORE julia starts.
export OMP_NUM_THREADS="$threads"
export OMP_CANCELLATION="TRUE"      # mandatory: SPRAL errors without it
export OMP_PROC_BIND="TRUE"         # pin threads to cores
export OPENBLAS_NUM_THREADS="2"     # avoid BLAS/SPRAL thread oversubscription

echo "Launching thermal optimization with OMP_NUM_THREADS=$threads"
julia --project=. scripts/3D/optimize_thermal_3d.jl