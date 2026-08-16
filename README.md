# Atomove
This repository was established in the process of doing my thesis titled "Optimal Control of Atomic Motion in Optical Tweezer Arrays" at the Niels Bohr Institute at the University of Copenhagen in order to finish my Master of Science in Quantum Information Science.

Julia code for computing optimal control pulses that move atoms between optical tweezer positions without heating them out of the trap. Written for my master's thesis.

The core idea: given start and target tweezer configurations, solve a trajectory optimization (via JuMP/Ipopt) that shapes the tweezer trap over time so the atom arrives with minimal residual motion/heating. There's a single-atom version, a thermal-ensemble version.

## Layout

- `src/` — physics models, basis functions, forward dynamics, and the optimal control solvers (`TweezerControls3D*.jl`)
- `scripts/` — entry points to run optimizations (`optimize_*_3d.jl`), forward simulation, and visualization

- `results/`, `ResultsForThesis/` — output data and figures
- `Archive/` — old (2D, pouring-simulation) code kept for reference, not maintained

## Running

```julia
julia --project=. scripts/optimize_single_3d.jl
```

Swap in `optimize_thermal_3d.jl` or `optimize_msa_3d.jl` for the other control problems. 

Requires Julia with the packages in `Project.toml` (JuMP, Ipopt, HDF5, etc. — instantiate with `julia --project=. -e 'using Pkg; Pkg.instantiate()'`).
