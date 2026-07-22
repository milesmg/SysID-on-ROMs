# ### ADJUSTED: Use a regular comment because Julia docstrings cannot attach to an `if` entrypoint block.
# Load the reusable visualization toolkit from a Julia notebook or REPL session.
if !isdefined(@__MODULE__, :VISUALIZATION_ROOT)
const VISUALIZATION_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

if !isdefined(@__MODULE__, :EquationSpec)
    for file in ("bootstrap.jl", "types.jl", "grids.jl", "laplacian.jl", "initial_conditions.jl",
                 "learners.jl", "losses.jl", "reduction.jl", "saving.jl", "cli.jl", "variable_windows.jl")
        include(joinpath(VISUALIZATION_ROOT, "src", "Core", file))
    end
    for file in ("allen_cahn.jl", "cahn_hilliard.jl", "reaction_diffusion.jl")
        include(joinpath(VISUALIZATION_ROOT, "src", "Equations", file))
    end
    include(joinpath(VISUALIZATION_ROOT, "src", "Core", "pipeline.jl"))
end

@eval using Plots

include(joinpath(@__DIR__, "run_data.jl"))
include(joinpath(@__DIR__, "trajectory_plots.jl"))
include(joinpath(@__DIR__, "rom_stability.jl"))
end
