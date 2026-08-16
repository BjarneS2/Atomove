# Atomove
This repository was created for my master's thesis, "Optimal Control of Atomic Motion in Optical Tweezer Arrays," completed at the Niels Bohr Institute, University of Copenhagen, in fulfillment of my MSc in Quantum Information Science.

It contains Julia code for computing optimal control pulses that move atoms between optical tweezer positions without heating them out of the trap.

The core idea: given start and target tweezer configurations, solve a trajectory optimization (via JuMP/Ipopt) that shapes the tweezer trap over time so the atom arrives with minimal residual motion/heating. There is a single-atom version and a thermal-ensemble version.

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

## AI declaration

Generative AI assistance (Claude Sonnet 4.6 and later on 5) was used during the development of this framework, limited to plotting and visualization scripts, code review, and bug finding/fixing to work more efficiently. All design and implementation choices were made by the author; the AI served only as a tool to increase efficiency and the code changes have all been monitored and manually approved. The optimizer is an advanced version of a 1D optimization framework the author previously developed during the thesis preparation project.
