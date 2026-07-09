const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))


include(joinpath(@__DIR__, "hpc_logging.jl"))

hpc_log("run_rom_hpc", "Julia entrypoint started")

### ADJUSTED: Read the learner flag before choosing the ROM implementation file.
function raw_cli_value(args, key, default)
    prefix = "--" * key * "="
    for i in eachindex(args)
        if args[i] == "--" * key && i < length(args)
            return args[i + 1]
        elseif startswith(args[i], prefix)
            return args[i][length(prefix)+1:end]
        end
    end
    return default
end

learner = lowercase(raw_cli_value(ARGS, "learner", raw_cli_value(ARGS, "model-type", "NN")))

hpc_log("run_rom_hpc", "Loading ROM helper code")
### ADJUSTED: Select the NN or polynomial ROM helper from the same runner.
hpc_log("run_rom_hpc", "Loading NN ROM helper code")
include(joinpath(REPO_ROOT, "Research_Code", "src", "HPC", "ROM_opt_AC_hpc.jl"))
if learner == "polynomial"
    hpc_log("run_rom_hpc", "Loading polynomial ROM extension code")
    include(joinpath(REPO_ROOT, "Research_Code", "src", "HPC", "FOM_ROM_polynomial_learning_AC.jl"))
end
hpc_log("run_rom_hpc", "Loading HPC common code")
include(joinpath(@__DIR__, "hpc_common.jl"))

opts = parse_cli(ARGS)
learner = lowercase(get_string(opts, "learner", learner))

N = get_int(opts, "N", 256)
L = get_float(opts, "L", 1.0)
ε2 = get_float(opts, "eps2", 1e-2)
k = get_float(opts, "k", 1.0)
### ADJUSTED: Parse spatial dimension for 1D/2D HPC ROM runs.
dimension = get_int(opts, "dimension", 1)
### ADJUSTED: Parse optional Allen-Cahn boundary-condition selection for ROM runs.
boundary_condition = get_string(opts, "boundary-condition", "homogeneous_dirichlet")
tfinal = get_float(opts, "tfinal", 2.0)
N_obs = get_int(opts, "N-obs", 10)
r = get_int(opts, "r", 10)
m = get_int(opts, "m", 10)
h = get_int(opts, "h", 8)
seed = get_int(opts, "seed", 1)
### ADJUSTED: Parse polynomial degree for polynomial ROM learning.
polynomial_degree = get_int(opts, "polynomial-degree", 3)
reference_dt_factor = get_float(opts, "reference-dt-factor", 0.5)
etas = get_float_vector(opts, "etas", [5e-3, 1e-3, 1e-4])
iterations = get_int_vector(opts, "iters", [500, 800, 1100])
### ADJUSTED: Parse variable-window schedules while defaulting every stage to the full trajectory.
stage_count = length(etas)
window_T = get_float_vector(opts, "window-T", fill(tfinal, stage_count))
window_N_obs = get_int_vector(opts, "window-N-obs", fill(N_obs, stage_count))
window_start_policy = get_string_vector(opts, "window-start-policy", fill("beginning", stage_count))
### ADJUSTED: Parse the trajectory-window batch size under its direct name.
batch_size = get_int_vector(opts, "batch-size", fill(1, stage_count))
loss_normalization = get_string(opts, "loss-normalization", "mean")
window_seed = get_int(opts, "window-seed", seed)
β = get_float_tuple(opts, "beta", (0.9, 0.99))
warmup = get_bool(opts, "warmup", true)
save_frequency = get_int(opts, "save-frequency", 10)
print_frequency = get_int(opts, "print-frequency", 10)
run_name = get_string(opts, "run-name", timestamp_run_name(learner == "polynomial" ? "ROM_AC_polynomial_hpc" : "ROM_AC_hpc"))

### ADJUSTED: Validate the complete staged learning and window schedule before building the reference solution.
schedule_lengths = (;
    etas=length(etas),
    iters=length(iterations),
    window_T=length(window_T),
    window_N_obs=length(window_N_obs),
    window_start_policy=length(window_start_policy),
    batch_size=length(batch_size),
)
all(==(stage_count), values(schedule_lengths)) || error("--etas, --iters, --window-T, --window-N-obs, --window-start-policy, and --batch-size must have the same length; got $schedule_lengths")
assert_run_name_available(run_name)

hpc_log("run_rom_hpc", "Configuring HPC runtime")
setup_hpc_runtime()

hpc_log("run_rom_hpc", "Building Allen-Cahn reference solution")
reference = build_ac_reference(; N, L, ε2, k, tfinal, reference_dt_factor, dimension, boundary_condition)
hpc_log("run_rom_hpc", "Building Laplacian matrix")
### ADJUSTED: Use the dimension- and boundary-aware sparse diffusion matrix for ROM basis construction.
A = get_lap_ac_matrix(N, ε2, reference.Δx, dimension, reference.boundary_condition)
hpc_log("run_rom_hpc", "Preparing ROM optimization problem")
### ADJUSTED: Pass polynomial degree only when the polynomial helper is active.
prob_rom = learner == "polynomial" ?
    prepare_ROM_optimization(A, reference.u_ref, r, m; N_obs, h, seed, dimension, boundary_condition=reference.boundary_condition, polynomial_degree) :
    prepare_ROM_optimization(A, reference.u_ref, r, m; N_obs, h, seed, dimension, boundary_condition=reference.boundary_condition)
hpc_log("run_rom_hpc", "Prepared ROM optimization parameters")
println("  run_name = ", run_name)
println("  learner = ", learner)
println("  polynomial_degree = ", learner == "polynomial" ? polynomial_degree : "n/a")
println("  N = ", N)
println("  L = ", L)
println("  dimension = ", dimension)
println("  boundary_condition = ", reference.boundary_condition)
println("  ε2 = ", ε2)
println("  k = ", k)
println("  tfinal = ", tfinal)
println("  N_obs = ", N_obs)
println("  r = ", r)
println("  m = ", m)
println("  h = ", h)
println("  seed = ", seed)
println("  reference_dt_factor = ", reference_dt_factor)
println("  reference_Δx = ", reference.Δx)
println("  reference_Δt = ", reference.Δt)
println("  reference_save_count = ", length(reference.t))
println("  etas = ", etas)
println("  iterations = ", iterations)
### ADJUSTED: Print the window schedule used by standard and sweep jobs.
println("  window_T = ", window_T)
println("  window_N_obs = ", window_N_obs)
println("  window_start_policy = ", window_start_policy)
println("  batch_size = ", batch_size)
println("  loss_normalization = ", loss_normalization)
println("  window_seed = ", window_seed)
println("  β = ", β)
println("  warmup = ", warmup)
println("  save_frequency = ", save_frequency)
println("  print_frequency = ", print_frequency)
println("  SLURM_JOB_ID = ", get(ENV, "SLURM_JOB_ID", "local"))
println("  SLURM_ARRAY_TASK_ID = ", get(ENV, "SLURM_ARRAY_TASK_ID", ""))
flush(stdout)
hpc_log("run_rom_hpc", "Running ROM optimization")
### ADJUSTED: Use the central ROM variable-window path for all runs, including full-trajectory defaults.
output = run_variable_window_ROM_optimization(
    reference.u_ref,
    prob_rom;
    eta=etas,
    beta=β,
    N_iter=iterations,
    window_T,
    window_N_obs,
    window_start_policy,
    batch_size,
    loss_normalization,
    window_seed,
    validation_N_obs=N_obs,
    warmup,
    save_frequency,
    print_frequency,
)
hpc_log("run_rom_hpc", "Saving ROM optimization data")
### ADJUSTED: Save window and full-trajectory validation histories for every ROM run.
save_dir = save_variable_window_ROM_optimization_data(output, run_name)

println("Saved ROM output to: ", save_dir)
println("Final full-trajectory loss: ", output.final_full_trajectory_loss)
println("Final training-window loss: ", output.final_training_loss)
flush(stdout)
