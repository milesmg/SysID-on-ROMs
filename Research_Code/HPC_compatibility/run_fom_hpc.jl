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
β = get_float_tuple(opts, "beta", (0.9, 0.99))
warmup = get_bool(opts, "warmup", true)
save_frequency = get_int(opts, "save-frequency", 10)
print_frequency = get_int(opts, "print-frequency", 10)
run_name = get_string(opts, "run-name", timestamp_run_name("FOM_AC_hpc"))

length(etas) == length(iterations) || error("--etas and --iters must have the same length")
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
    ),
)
hpc_log("run_fom_hpc", "Setting up Optimization.jl problem")
optprob = set_up_optimization(reference.u_ref, prepared.prob, prepared.t_obs, prepared.p₀; run_params)
hpc_log("run_fom_hpc", "Running FOM optimization")
output = run_full_optimization(optprob; η=etas, β, N_iter=iterations, warmup, save_frequency, print_frequency)
hpc_log("run_fom_hpc", "Saving FOM optimization data")
save_dir = save_optimization_data(output, run_name)

println("Saved FOM output to: ", save_dir)
println("Final loss: ", output.final_loss)
flush(stdout)
