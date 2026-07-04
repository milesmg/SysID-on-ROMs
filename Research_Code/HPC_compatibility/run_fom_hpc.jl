const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "hpc_logging.jl"))

hpc_log("run_fom_hpc", "Julia entrypoint started")

hpc_log("run_fom_hpc", "Loading integration helper code")
include(joinpath(REPO_ROOT, "Research_Code", "helper_functions","HPC", "integration_AC_hpc.jl"))
hpc_log("run_fom_hpc", "Loading FOM helper code")
include(joinpath(REPO_ROOT, "Research_Code", "helper_functions","HPC", "FOM_opt_AC_hpc.jl"))
hpc_log("run_fom_hpc", "Loading HPC common code")
include(joinpath(@__DIR__, "hpc_common.jl"))

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
stage_count = length(etas)
window_T = get_float_vector(opts, "window-T", fill(tfinal, stage_count))
window_N_obs = get_int_vector(opts, "window-N-obs", fill(N_obs, stage_count))
window_start_policy = get_string_vector(opts, "window-start-policy", fill("beginning", stage_count))
windows_per_iter = get_int_vector(opts, "windows-per-iter", fill(1, stage_count))
loss_normalization = get_string(opts, "loss-normalization", "mean")
window_seed = get_int(opts, "window-seed", seed)
β = get_float_tuple(opts, "beta", (0.9, 0.99))
warmup = get_bool(opts, "warmup", true)
save_frequency = get_int(opts, "save-frequency", 10)
print_frequency = get_int(opts, "print-frequency", 10)
run_name = get_string(opts, "run-name", timestamp_run_name("FOM_AC_hpc"))

schedule_lengths = (;
    etas=length(etas),
    iters=length(iterations),
    window_T=length(window_T),
    window_N_obs=length(window_N_obs),
    window_start_policy=length(window_start_policy),
    windows_per_iter=length(windows_per_iter),
)
all(==(stage_count), values(schedule_lengths)) || error("--etas, --iters, --window-T, --window-N-obs, --window-start-policy, and --windows-per-iter must have the same length; got $schedule_lengths")
assert_run_name_available(run_name)

hpc_log("run_fom_hpc", "Configuring HPC runtime")
setup_hpc_runtime()

hpc_log("run_fom_hpc", "Building Allen-Cahn reference solution")
reference = build_ac_reference(; N, L, ε2, k, tfinal, reference_dt_factor)
hpc_log("run_fom_hpc", "Preparing FOM optimization problem")
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
hpc_log("run_fom_hpc", "Prepared FOM optimization parameters")
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
println("  window_T = ", window_T)
println("  window_N_obs = ", window_N_obs)
println("  window_start_policy = ", window_start_policy)
println("  windows_per_iter = ", windows_per_iter)
println("  loss_normalization = ", loss_normalization)
println("  window_seed = ", window_seed)
println("  β = ", β)
println("  warmup = ", warmup)
println("  save_frequency = ", save_frequency)
println("  print_frequency = ", print_frequency)
println("  SLURM_JOB_ID = ", get(ENV, "SLURM_JOB_ID", "local"))
println("  SLURM_ARRAY_TASK_ID = ", get(ENV, "SLURM_ARRAY_TASK_ID", ""))
flush(stdout)
hpc_log("run_fom_hpc", "Running FOM optimization")
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
hpc_log("run_fom_hpc", "Saving FOM optimization data")
save_dir = save_variable_window_optimization_data(output, run_name)

println("Saved FOM output to: ", save_dir)
println("Final full-trajectory loss: ", output.final_full_trajectory_loss)
println("Final training-window loss: ", output.final_training_loss)
flush(stdout)
