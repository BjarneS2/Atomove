param([int]$threads = 7)

# SPRAL (Ipopt's parallel linear solver) reads these at DLL load time,
# so they must be set BEFORE julia starts.
$env:OMP_NUM_THREADS    = "$threads"
$env:OMP_CANCELLATION   = "TRUE"    # mandatory: SPRAL errors without it
$env:OMP_PROC_BIND      = "TRUE"    # pin threads to cores
$env:OPENBLAS_NUM_THREADS = "1"     # avoid BLAS/SPRAL thread oversubscription

Write-Host "Launching thermal optimization with OMP_NUM_THREADS=$threads"
julia --project=. scripts/3D/optimize_thermal_3d.jl