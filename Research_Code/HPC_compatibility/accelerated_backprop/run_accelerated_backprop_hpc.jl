### ADJUSTED: Mirror the variable-window FOM entrypoint while exposing backprop benchmark choices.
include(joinpath(@__DIR__, "accelerated_backprop_FOM_hpc.jl"))

function get_string_vector(opts, key, default)
    return strip.(split(get(opts, key, join(default, ",")), ","))
end

opts = parse_cli(ARGS)

N = get_int(opts, "N", 256)
L = get_float(opts, "L", 1.0)
ε2 = get_float(opts, "eps2", 1e-2)
k = get_float(opts, "k", 1.0)
tfinal = get_float(opts, "tfinal", 2.0)
N_obs = get_int(opts, "N-obs", 100)
h = get_int(opts, "h", 8)
seed = get_int(opts, "seed", 1)
reference_dt_factor = get_float(opts, "reference-dt-factor", 0.5)
networks = get_string_vector(opts, "networks", ["lux", "simplechains"])
vjps = get_string_vector(opts, "vjps", collect(BACKPROP_VJP_NAMES))
window_T_values = get_float_vector(opts, "window-T", [0.1, 2.0])
window_N_obs_values = get_int_vector(opts, "window-N-obs", [10, 50])
repeats = get_int(opts, "repeats", 3)
directional_step = get_float(opts, "directional-step", 1e-4)
solver_autodiff = get_string(opts, "solver-autodiff", "finite_diff")
run_name = get_string(opts, "run-name", timestamp_run_name("accelerated_backprop"))
data_root = get_string(opts, "data-root", normpath(joinpath(@__DIR__, "..", "..", "Optimization", "Data", "BackpropBenchmarks")))

setup_hpc_runtime()

### ADJUSTED: Reuse the production FOM reference and network preparation unchanged.
reference = build_ac_reference(; N, L, ε2, k, tfinal, reference_dt_factor)
prepared = prepare_for_optimization(;
    N,
    L,
    ε2,
    tspan=reference.tspan,
    N_obs,
    u₀=reference.u₀,
    h,
    seed,
)

println("Accelerated-backprop benchmark parameters:")
println("  run_name = ", run_name)
println("  N = ", N)
println("  h = ", h)
println("  networks = ", networks)
println("  vjps = ", vjps)
println("  window_T = ", window_T_values)
println("  window_N_obs = ", window_N_obs_values)
println("  repeats = ", repeats)
println("  directional_step = ", directional_step)
println("  solver_autodiff = ", solver_autodiff)
flush(stdout)

output = run_accelerated_backprop_benchmarks(
    reference,
    prepared;
    networks,
    vjps,
    window_T_values,
    window_N_obs_values,
    repeats,
    directional_step,
    solver_autodiff,
    run_name,
    data_root,
)

### ADJUSTED: Known package incompatibilities are results; only unexpected failures fail the job.
failed = count(row -> row.status == "error", output.rows)
unsupported = count(row -> row.status == "unsupported", output.rows)
println("Saved benchmark output to: ", output.run_directory)
println("Successful combinations: ", count(row -> row.status == "ok", output.rows))
println("Unsupported combinations: ", unsupported)
println("Unexpected failures: ", failed)
flush(stdout)

failed == 0 || exit(1)
