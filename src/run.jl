### ADJUSTED: Use one top-level entrypoint for FOM and ROM training.
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

for file in ("bootstrap.jl", "types.jl", "grids.jl", "laplacian.jl", "initial_conditions.jl",
             "learners.jl", "losses.jl", "reduction.jl", "saving.jl", "cli.jl", "variable_windows.jl")
    include(joinpath(REPO_ROOT, "src", "Core", file))
end
for file in ("allen_cahn.jl", "cahn_hilliard.jl", "reaction_diffusion.jl")
    include(joinpath(REPO_ROOT, "src", "Equations", file))
end
include(joinpath(REPO_ROOT, "src", "Core", "pipeline.jl"))

length(ARGS) >= 2 && ARGS[1] == "--mode" || error("first arguments must be --mode fom or --mode rom")
mode = Symbol(ARGS[2])
mode in (:fom, :rom) || error("mode must be fom or rom")
run_training(mode, ARGS[3:end])
