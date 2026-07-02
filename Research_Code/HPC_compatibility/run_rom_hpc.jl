const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))


include(joinpath(@__DIR__, "hpc_logging.jl"))

hpc_log("run_rom_hpc", "Julia entrypoint started")


hpc_log("run_rom_hpc", "Loading ROM helper code")
include(joinpath(REPO_ROOT, "Research_Code", "helper_functions", "HPC", "ROM_opt_AC_hpc.jl"))
hpc_log("run_rom_hpc", "Loading HPC common code")
include(joinpath(@__DIR__, "hpc_common.jl"))

opts = parse_cli(ARGS)

N = get_int(opts, "N", 256)
L = get_float(opts, "L", 1.0)
ε2 = get_float(opts, "eps2", 1e-2)
k = get_float(opts, "k", 1.0)
tfinal = get_float(opts, "tfinal", 2.0)
N_obs = get_int(opts, "N-obs", 10)
r = get_int(opts, "r", 10)
m = get_int(opts, "m", 10)
h = get_int(opts, "h", 8)
seed = get_int(opts, "seed", 1)
reference_dt_factor = get_float(opts, "reference-dt-factor", 0.5)
etas = get_float_vector(opts, "etas", [5e-3, 1e-3, 1e-4])
iterations = get_int_vector(opts, "iters", [500, 800, 1100])
β = get_float_tuple(opts, "beta", (0.9, 0.99))
warmup = get_bool(opts, "warmup", true)
save_frequency = get_int(opts, "save-frequency", 10)
print_frequency = get_int(opts, "print-frequency", 10)
run_name = get_string(opts, "run-name", timestamp_run_name("ROM_AC_hpc"))

length(etas) == length(iterations) || error("--etas and --iters must have the same length")
assert_run_name_available(run_name)

hpc_log("run_rom_hpc", "Configuring HPC runtime")
setup_hpc_runtime()

hpc_log("run_rom_hpc", "Building Allen-Cahn reference solution")
reference = build_ac_reference(; N, L, ε2, k, tfinal, reference_dt_factor)
hpc_log("run_rom_hpc", "Building Laplacian matrix")
A = get_lap1d_matrix(N, ε2, reference.Δx)
hpc_log("run_rom_hpc", "Preparing ROM optimization problem")
prob_rom = prepare_ROM_optimization(A, reference.u_ref, r, m; N_obs, h, seed)
hpc_log("run_rom_hpc", "Setting up Optimization.jl problem")
optprob_rom = set_up_ROM_optimization(prob_rom)
hpc_log("run_rom_hpc", "Running ROM optimization")
output = run_ROM_optimization(optprob_rom; η=etas, β, N_iter=iterations, warmup, save_frequency, print_frequency)
hpc_log("run_rom_hpc", "Saving ROM optimization data")
save_dir = save_ROM_optimization_data(output, run_name)

println("Saved ROM output to: ", save_dir)
println("Final loss: ", output.final_loss)
flush(stdout)
