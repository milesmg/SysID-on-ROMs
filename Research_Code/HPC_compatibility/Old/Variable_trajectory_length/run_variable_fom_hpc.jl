
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(@__DIR__, "..", "hpc_logging.jl"))

hpc_log("run_variable_fom_hpc", "Julia entrypoint started")

function parse_cli_light(args)
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

light_get_float_vector(opts, key, default) = parse.(Float64, split(get(opts, key, join(default, ",")), ","))
light_get_int_vector(opts, key, default) = parse.(Int, split(get(opts, key, join(default, ",")), ","))
light_get_string_vector(opts, key, default) = strip.(split(get(opts, key, join(default, ",")), ","))
light_get_float(opts, key, default) = parse(Float64, get(opts, key, string(default)))
light_get_int(opts, key, default) = parse(Int, get(opts, key, string(default)))

light_opts = parse_cli_light(ARGS)
light_tfinal = light_get_float(light_opts, "tfinal", 2.0)
light_N_obs = light_get_int(light_opts, "N-obs", 10)
light_schedule_lengths = (;
    etas=length(light_get_float_vector(light_opts, "etas", [5e-3, 1e-3, 1e-4])),
    iters=length(light_get_int_vector(light_opts, "iters", [500, 800, 1100])),
    window_T=length(light_get_float_vector(light_opts, "window-T", [light_tfinal])),
    window_N_obs=length(light_get_int_vector(light_opts, "window-N-obs", [light_N_obs])),
    window_start_policy=length(light_get_string_vector(light_opts, "window-start-policy", ["beginning"])),
    windows_per_iter=length(light_get_int_vector(light_opts, "windows-per-iter", [1])),
)
all(==(light_schedule_lengths.etas), values(light_schedule_lengths)) || error("--etas, --iters, --window-T, --window-N-obs, --window-start-policy, and --windows-per-iter must have the same length; got $light_schedule_lengths")

hpc_log("run_variable_fom_hpc", "Loading variable-window FOM helper code")
include(joinpath(@__DIR__, "variable_window_FOM_opt_AC_hpc.jl"))
hpc_log("run_variable_fom_hpc", "Loading HPC common code")
include(joinpath(@__DIR__, "..", "hpc_common.jl"))


function get_string_vector(opts, key, default)
    raw = get(opts, key, join(default, ","))
    return strip.(split(raw, ","))
end

opts = parse_cli(ARGS)

N = get_int(opts, "N", 256)
L = get_float(opts, "L", 1.0)
ε2 = get_float(opts, "eps2", 1e-2)
k = get_float(opts, "k", 1.0)
tfinal = get_float(opts, "tfinal", 2.0)
N_obs = get_int(opts, "N-obs", 10)
h = get_int(opts, "h", 8)
seed = get_int(opts, "seed", 1)
reference_dt_factor = get_float(opts, "reference-dt-factor", 0.5)
etas = get_float_vector(opts, "etas", [5e-3, 1e-3, 1e-4])
iterations = get_int_vector(opts, "iters", [500, 800, 1100])
β = get_float_tuple(opts, "beta", (0.9, 0.99))
warmup = get_bool(opts, "warmup", true)
save_frequency = get_int(opts, "save-frequency", 10)
print_frequency = get_int(opts, "print-frequency", 10)
window_T = get_float_vector(opts, "window-T", [tfinal])
window_N_obs = get_int_vector(opts, "window-N-obs", [N_obs])
window_start_policy = get_string_vector(opts, "window-start-policy", ["beginning"])
windows_per_iter = get_int_vector(opts, "windows-per-iter", [1])
loss_normalization = get_string(opts, "loss-normalization", "mean")
window_seed = get_int(opts, "window-seed", seed)
run_name = get_string(opts, "run-name", timestamp_run_name("FOM_variable_window_hpc"))

schedule_lengths = (;
    etas=length(etas),
    iters=length(iterations),
    window_T=length(window_T),
    window_N_obs=length(window_N_obs),
    window_start_policy=length(window_start_policy),
    windows_per_iter=length(windows_per_iter),
)
all(==(schedule_lengths.etas), values(schedule_lengths)) || error("--etas, --iters, --window-T, --window-N-obs, --window-start-policy, and --windows-per-iter must have the same length; got $schedule_lengths")
assert_run_name_available(run_name)

hpc_log("run_variable_fom_hpc", "Configuring HPC runtime")
setup_hpc_runtime()

hpc_log("run_variable_fom_hpc", "Building Allen-Cahn reference solution")
reference = build_ac_reference(; N, L, ε2, k, tfinal, reference_dt_factor)
hpc_log("run_variable_fom_hpc", "Preparing FOM optimization problem")
prepared = prepare_for_optimization(; N, L, ε2, tspan=reference.tspan, N_obs, u₀=reference.u₀, h, seed)
run_params = merge(
    prepared.run_params,
    (;
        k,
        reference_dt=reference.Δt,
        reference_dt_factor,
        reference_save_count=length(reference.t),
        hpc_job_id=get(ENV, "SLURM_JOB_ID", "local"),
        hpc_array_task_id=get(ENV, "SLURM_ARRAY_TASK_ID", ""),
        variable_window_training=true,
    ),
)

hpc_log("run_variable_fom_hpc", "Prepared variable-window FOM optimization parameters")
println("  run_name = ", run_name)
println("  N = ", N)
println("  L = ", L)
println("  ε2 = ", ε2)
println("  k = ", k)
println("  tfinal = ", tfinal)
println("  N_obs = ", N_obs)
println("  h = ", h)
println("  seed = ", seed)
println("  reference_dt_factor = ", reference_dt_factor)
println("  reference_Δx = ", reference.Δx)
println("  reference_Δt = ", reference.Δt)
println("  reference_save_count = ", length(reference.t))
println("  etas = ", etas)
println("  iterations = ", iterations)
println("  β = ", β)
println("  warmup = ", warmup)
println("  save_frequency = ", save_frequency)
println("  print_frequency = ", print_frequency)
println("  window_T = ", window_T)
println("  window_N_obs = ", window_N_obs)
println("  window_start_policy = ", window_start_policy)
println("  windows_per_iter = ", windows_per_iter)
println("  loss_normalization = ", loss_normalization)
println("  window_seed = ", window_seed)
println("  SLURM_JOB_ID = ", get(ENV, "SLURM_JOB_ID", "local"))
println("  SLURM_ARRAY_TASK_ID = ", get(ENV, "SLURM_ARRAY_TASK_ID", ""))
flush(stdout)

hpc_log("run_variable_fom_hpc", "Running variable-window FOM optimization")
output = run_variable_window_optimization(
    reference.u_ref,
    prepared.prob,
    prepared.p₀;
    run_params,
    eta=etas,
    beta=β,
    N_iter=iterations,
    window_T,
    window_N_obs,
    window_start_policy,
    windows_per_iter,
    loss_normalization,
    window_seed,
    validation_N_obs=N_obs,
    warmup,
    save_frequency,
    print_frequency,
)
hpc_log("run_variable_fom_hpc", "Saving variable-window FOM optimization data")
save_dir = save_variable_window_optimization_data(output, run_name)

println("Saved variable-window FOM output to: ", save_dir)
println("Final full-trajectory loss: ", output.final_full_trajectory_loss)
println("Final training-window loss: ", output.final_training_loss)
flush(stdout)
