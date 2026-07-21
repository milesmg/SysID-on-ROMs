### Load necessary packages

### ADJUSTED: Load shared logging from the top-level tooling directory.
include(joinpath(@__DIR__, "..", "Tools", "Misc", "logging.jl"))

for package in ("LinearAlgebra", "SparseArrays", "Statistics", "Random",
                "ComponentArrays", "LinearSolve", "OrdinaryDiffEq",
                "OrdinaryDiffEqSDIRK", "OrdinaryDiffEqLowOrderRK",
                "SciMLSensitivity", "ADTypes", "Zygote", "Mooncake",
                "Optimization", "OptimizationOptimisers", "OptimizationOptimJL",
                "LineSearches", "Lux", "Functors", "Dates", "Serialization")
    hpc_log_package(package, "Loading")
    # @eval for Julia runtime issue
    @eval using $(Symbol(package))
    hpc_log_package(package, "Loaded")
end
