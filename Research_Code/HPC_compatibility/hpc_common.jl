include(joinpath(@__DIR__, "hpc_logging.jl"))
include(joinpath(@__DIR__, "..", "src", "Misc.", "run_name_guard.jl"))

hpc_log_package("LinearAlgebra", "Loading")
using LinearAlgebra
hpc_log_package("LinearAlgebra", "Loaded")
hpc_log_package("SparseArrays", "Loading")
using SparseArrays
hpc_log_package("SparseArrays", "Loaded")
hpc_log_package("ComponentArrays", "Loading")
using ComponentArrays
hpc_log_package("ComponentArrays", "Loaded")
hpc_log_package("OrdinaryDiffEq", "Loading")
using OrdinaryDiffEq
hpc_log_package("OrdinaryDiffEq", "Loaded")
hpc_log_package("OrdinaryDiffEqLowOrderRK", "Loading")
using OrdinaryDiffEqLowOrderRK
hpc_log_package("OrdinaryDiffEqLowOrderRK", "Loaded")
hpc_log_package("Optimization", "Loading")
using Optimization
hpc_log_package("Optimization", "Loaded")
hpc_log_package("Dates", "Loading")
using Dates
hpc_log_package("Dates", "Loaded")

"""
Parse named command-line flags into a string dictionary.

- Accepts `--key value`, `--key=value`, and boolean `--key` forms.
- Positional arguments are ignored.
"""
function parse_cli(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key_value = arg[3:end]
            if occursin("=", key_value)
                key, value = split(key_value, "="; limit=2)
                opts[key] = value
            elseif i < length(args) && !startswith(args[i + 1], "--")
                opts[key_value] = args[i + 1]
                i += 1
            else
                opts[key_value] = "true"
            end
        end
        i += 1
    end
    return opts
end

"""Return a string option from `opts`, or `default` if it is absent."""
get_string(opts, key, default) = get(opts, key, default)

"""Return an integer option from `opts`, or `default` if it is absent."""
get_int(opts, key, default) = parse(Int, get(opts, key, string(default)))

"""Return a floating-point option from `opts`, or `default` if it is absent."""
get_float(opts, key, default) = parse(Float64, get(opts, key, string(default)))

"""Return a boolean option from `opts`, accepting true/1/yes/y as true values."""
get_bool(opts, key, default) = lowercase(get(opts, key, string(default))) in ("true", "1", "yes", "y")

"""Return a comma-separated floating-point tuple option from `opts`."""
get_float_tuple(opts, key, default) = Tuple(parse.(Float64, split(get(opts, key, join(default, ",")), ",")))

"""Return a comma-separated floating-point vector option from `opts`."""
get_float_vector(opts, key, default) = parse.(Float64, split(get(opts, key, join(default, ",")), ","))

"""Return a comma-separated integer vector option from `opts`."""
get_int_vector(opts, key, default) = parse.(Int, split(get(opts, key, join(default, ",")), ","))

### ADJUSTED: Parse staged window policies for the standard FOM and ROM runners.
"""Return a comma-separated string vector option from `opts`."""
get_string_vector(opts, key, default) = strip.(split(get(opts, key, join(default, ",")), ","))

"""
Configure and print runtime information for hpc1 batch jobs.

BLAS threads are chosen from `JULIA_BLAS_THREADS`, then default to `1`
for the measured J8/B1 HPC training setup.
"""
function setup_hpc_runtime()
    ### ADJUSTED: Keep BLAS single-threaded unless explicitly overridden.
    blas_threads = parse(Int, get(ENV, "JULIA_BLAS_THREADS", "1"))
    BLAS.set_num_threads(blas_threads)
    println("Julia version: ", VERSION)
    println("Host: ", get(ENV, "HOSTNAME", "unknown"))
    println("SLURM_JOB_ID: ", get(ENV, "SLURM_JOB_ID", "not_slurm"))
    println("SLURM_ARRAY_TASK_ID: ", get(ENV, "SLURM_ARRAY_TASK_ID", "none"))
    println("Julia threads: ", Threads.nthreads())
    println("BLAS threads: ", BLAS.get_num_threads())
end

"""
Create a run name with a timestamp and Slurm job/task suffix.
"""
function timestamp_run_name(prefix)
    job = get(ENV, "SLURM_JOB_ID", "local")
    task = get(ENV, "SLURM_ARRAY_TASK_ID", "")
    suffix = isempty(task) ? job : "$(job)_$(task)"
    return "$(prefix)_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))_$(suffix)"
end

"""
Build the Allen-Cahn reference solution used by FOM and ROM training.

- args: `N`, `L`, `ε2`, `k`, `tfinal`, `reference_dt_factor`, `dimension`, `boundary_condition`
- returns: named tuple with `u_ref`, `prob`, `p₀`, `u₀`, `x`, `t`, `tspan`, `Δx`, and `Δt`
"""
function build_ac_reference(; N, L, ε2, k, tfinal, reference_dt_factor, dimension=1, boundary_condition="homogeneous_dirichlet", u₀=nothing)
    dimension = validate_ac_dimension(dimension)
    boundary_condition = validate_ac_boundary_condition(boundary_condition)
    grid = ac_grid(N, L, boundary_condition)
    Δx = grid.Δx
    x = grid.x
    u₀ = isnothing(u₀) ?
        default_ac_initial_condition(N, L, ε2, dimension, boundary_condition) :
        normalize_ac_initial_condition(u₀, N, dimension)
    tspan = (0.0, tfinal)
    Δt = reference_dt_factor * Δx^2 / (2 * dimension * ε2)
    max_saved_time_count = 500
    saved_time_count = min(
        max_saved_time_count,
        max(2, floor(Int, (tspan[2] - tspan[1]) / Δt) + 1),
    )
    t = LinRange(tspan[1], tspan[2], saved_time_count)
    p₀ = (; ε2, k, Δx, N, dimension, boundary_condition)
    jac_prototype = dimension == 1 && boundary_condition == "homogeneous_dirichlet" ?
        Tridiagonal(zeros(N - 1), zeros(N), zeros(N - 1)) :
        get_lap_ac_matrix(N, 1.0, 1.0, dimension, boundary_condition)
    f = ODEFunction(rhs_ac!; jac_prototype)
    prob = ODEProblem(f, u₀, tspan, p₀)
    println("Reference solve: dimension=$dimension, boundary_condition=$boundary_condition, N=$N, state_length=$(length(u₀)), Δx=$Δx, Δt=$Δt, saved_times=$(length(t))")
    u_ref = solve(prob, Euler(); dt=Δt, saveat=t)
    return (; u_ref, prob, p₀, u₀, x, y=dimension == 1 ? nothing : copy(x), t, tspan, Δx, Δt, dimension, boundary_condition, state_shape=dimension == 1 ? (N,) : (N, N))
end
