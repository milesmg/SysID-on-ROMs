# Parse Command Line arguments

"""Parse a list of command line arguments
    - input is Julia's built-in Vector{String} with the command line arguments;
        - each entry in the vector is from space- or line break-separated string
    - builds a dictionary of form (arg, val)
    - adjusts for --, =, etc"""
function parse_cli(args)
    options = Dict{String, String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key_value = arg[3:end]
            if occursin("=", key_value)
                key, value = split(key_value, "="; limit=2)
                options[key] = value
            elseif i < length(args) && !startswith(args[i + 1], "--")
                options[key_value] = args[i + 1]
                i += 1
            else
                options[key_value] = "true"
            end
        end
        i += 1
    end
    options
end

# get is Julia's dictionary lookup function. 
# options is the dictionary; key is what it's looking for; default is what it will default to 
get_string(options, key, default) = get(options, key, default)
get_int(options, key, default) = parse(Int, get(options, key, string(default)))
get_float(options, key, default) = parse(Float64, get(options, key, string(default)))
get_bool(options, key, default) = lowercase(get(options, key, string(default))) in ("true", "1", "yes", "y")
# get a string, add a comma, split by commas, parse element wise for floats, convert to tuple
get_float_tuple(options, key, default) = Tuple(parse.(Float64, split(get(options, key, join(default, ",")), ",")))
get_float_vector(options, key, default) = parse.(Float64, split(get(options, key, join(default, ",")), ","))
get_int_vector(options, key, default) = parse.(Int, split(get(options, key, join(default, ",")), ","))
get_string_vector(options, key, default) = strip.(split(get(options, key, join(default, ",")), ","))

"""
Set threading for the optimization and print results
    - Default is 1 BLAS thread
    - Most of these variables/objects (BLAS, ENV, Threads, etc.) are built in to Julia
"""
function setup_runtime()
    BLAS.set_num_threads(get_int(ENV, "JULIA_BLAS_THREADS", 1))
    println("Julia version: ", VERSION)
    println("Host: ", get_string(ENV, "HOSTNAME", "unknown"))
    println("SLURM_JOB_ID: ", get_string(ENV, "SLURM_JOB_ID", "not_slurm"))
    println("Julia threads: ", Threads.nthreads())
    println("BLAS threads: ", BLAS.get_num_threads())
end

"""
Builds a timestamped run name based on the Job ID, and, if it exists, the Task ID, stored in ENV
- note that they're stored in the global ENV, which is inherited by the child process Julia, automatically by slurm
"""
timestamp_run_name(prefix) = begin
    job = get_string(ENV, "SLURM_JOB_ID", "local")
    task = get_string(ENV, "SLURM_ARRAY_TASK_ID", "")
    "$(prefix)_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))_$(isempty(task) ? job : "$(job)_$(task)")"
end

"""
Sets the parameters for a run based on dictionary of command line arguments. 
    - etas default: None! Must be passed in
    - iterations default: None! Must be passed in
    - stage count default = length(etas)
    - window_T default = tfinal, for each stage
    - window_N_obs default = global Nobs (), for each stage [this N_obs builds the validation ]
    - window-start-policy default = beginning, for each stage
    - loss_normalization default = mean
    - loss_space default = FULL
    - window_seed default = seed, passed in as argument
    - beta default (0.0,0.99)
    - warmup default = true
    - save_frequency default = 10
    - print-frequency default = 10 
    - learned_function_error defaults to false
    - learned_function_error_bounds defaults to (-1.0,1.0)
    - NOTE: This is where we validate that the number of stages is the same across inputs
"""
function parse_training_options(options, tfinal, N_obs, seed)::TrainingConfig
    haskey(options, "etas") || error("There is no default learn rate schedule")
    etas = get_float_vector(options, "etas", Float64[])
    haskey(options, "iters") || error("There is no default iteration schedule")
    iterations = get_int_vector(options, "iters", Int[])
    stage_count = length(etas)
    window_T = get_float_vector(options, "window-T", fill(tfinal, stage_count))
    window_N_obs = get_int_vector(options, "window-N-obs", fill(N_obs, stage_count))
    window_start_policy = get_string_vector(options, "window-start-policy", fill("beginning", stage_count))
    loss_space = uppercase(get_string(options, "loss-space", "FULL"))
    loss_space in ("FULL", "REDUCED") || error("loss-space must be FULL or REDUCED")
    learned_function_error_bounds = get_float_tuple(options, "learned-function-error-bounds", (-1.0, 1.0))
    lengths = (length(iterations), length(window_T), length(window_N_obs),
               length(window_start_policy))
    all(==(stage_count), lengths) || error("staged schedules must have equal lengths")
    TrainingConfig(etas, iterations, window_T, window_N_obs, window_start_policy,
                   get_string(options, "loss-normalization", "mean"), loss_space,
                   get_int(options, "window-seed", seed),
                   get_float_tuple(options, "beta", (0.9, 0.99)),
                   get_bool(options, "warmup", true),
                   get_int(options, "save-frequency", 10),
                   get_int(options, "print-frequency", 10),
                   get_bool(options, "learned-function-error", false),
                   learned_function_error_bounds)
end
