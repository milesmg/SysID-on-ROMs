# Containment map:
# EquationSpec -> equation defaults and adapter functions
# RunConfig -> EquationParameters
# ReferenceData -> EquationParameters
# PreparedTraining -> RunConfig, Grid, LearnerSetup, ReferenceData, ROMData or nothing
# TrainingWindow -> WindowSpec
# TrainingOutput -> TrainingConfig, TrainingSnapshot, WindowSpec

struct EquationParameters
    ε2::Float64 # Allen-Cahn or Cahn-Hilliard interface parameter
    k::Float64 # Allen-Cahn cubic-reaction coefficient
    sigma::Float64 # Cahn-Hilliard linear relaxation coefficient
    mean_c::Float64 # Cahn-Hilliard conserved mean state
    D1::Float64 # reaction-diffusion first-species diffusivity
    D2::Float64 # reaction-diffusion second-species diffusivity
    r::Int # requested state POD rank
    m::Int # requested nonlinear DEIM rank
    forced_deim_split::Bool # split reaction-diffusion DEIM points evenly by reaction function
end
# Appears in:
# - Core/types.jl
# - Core/pipeline.jl
# - Core/saving.jl
# - Equations/allen_cahn.jl
# - Equations/cahn_hilliard.jl
# - Equations/reaction_diffusion.jl
# - Tools/Tests/runtests.jl

EquationParameters(; ε2=0.0, k=0.0, sigma=0.0, mean_c=0.0, D1=0.0, D2=0.0, r=0, m=0, forced_deim_split=false) =
    EquationParameters(Float64(ε2), Float64(k), Float64(sigma), Float64(mean_c), Float64(D1), Float64(D2), Int(r), Int(m), Bool(forced_deim_split))

struct Grid
    N::Int # grid points along each spatial dimension
    L::Float64 # physical domain length
    dimension::Int # one or two spatial dimensions
    boundary_condition::String # canonical boundary condition
    Δx::Float64 # uniform spatial spacing
    x::Vector{Float64} # coordinates along the first spatial axis
    y::Union{Nothing,Vector{Float64}} # coordinates along the second spatial axis
    state_shape::Tuple{Vararg{Int}} # unflattened single-field state dimensions
end
# Appears in:
# - Core/types.jl
# - Core/grids.jl
# - Core/initial_conditions.jl
# - Core/laplacian.jl
# - Core/losses.jl
# - Core/pipeline.jl
# - Equations/allen_cahn.jl
# - Equations/cahn_hilliard.jl
# - Equations/reaction_diffusion.jl

struct RunConfig
    N::Int # grid points along each spatial dimension
    L::Float64 # physical domain length
    tfinal::Float64 # final physical time
    N_obs::Int # full-trajectory validation observations
    h::Int # hidden-layer width for neural learners
    seed::Int # learner and reference random seed
    dimension::Int # spatial dimension
    boundary_condition::String # canonical boundary condition
    learner::String # learner family name
    polynomial_degree::Int # polynomial learner degree
    reference_dt_factor::Float64 # reference-solver time-step multiplier
    initial_condition::String # selected initial-condition name
    parameters::EquationParameters # equation-specific physical and ROM parameters
end
# Appears in:
# - Core/types.jl
# - Core/initial_conditions.jl
# - Core/pipeline.jl
# - Equations/allen_cahn.jl
# - Equations/cahn_hilliard.jl
# - Equations/reaction_diffusion.jl

struct EquationSpec
    name::String # equation selector used by runners and metadata
    fields::Int # number of state fields
    input_dim::Int # learner input dimension
    default_N::Int # default grid points along each spatial dimension
    default_tfinal::Float64 # default final physical time
    default_dimension::Int # default spatial dimension
    default_boundary_condition::String # default canonical boundary condition
    parse_parameters::Function # CLI parser returning EquationParameters
    default_initial_condition::Function # default-state materializer
    named_initial_condition::Function # named-state materializer
    reference::Function # reference-solve builder
    model::Function # FOM or ROM model builder
end
# Appears in:
# - Core/types.jl
# - Core/initial_conditions.jl
# - Core/pipeline.jl
# - Equations/allen_cahn.jl
# - Equations/cahn_hilliard.jl
# - Equations/reaction_diffusion.jl

struct LearnerSetup
    kind::String # neural-network or polynomial learner family
    nn::Any # Lux neural-network object when applicable
    state::Any # Lux nontrainable state when applicable
    θ::Any # initial trainable parameters or polynomial coefficients
    h::Union{Nothing,Int} # neural-network hidden width
    seed::Int # learner initialization seed
    polynomial_degree::Union{Nothing,Int} # polynomial degree when applicable
    activation::String # learner activation or polynomial label
end
# Appears in:
# - Core/types.jl
# - Core/learners.jl
# - Core/pipeline.jl
# - Equations/allen_cahn.jl
# - Equations/cahn_hilliard.jl
# - Equations/reaction_diffusion.jl

struct ReferenceData{S,P,O}
    solution::S # full-order reference ODE solution
    problem::P # full-order reference ODE problem
    parameters::EquationParameters # physical parameters used by the reference RHS
    initial_state::Vector{Float64} # flattened full-order initial state
    operator::O # sparse operator retained for Jacobians and ROM construction
    times::Vector{Float64} # saved reference times
    tspan::Tuple{Float64,Float64} # reference integration interval
    Δt::Float64 # reference integration step size
    N_obs::Int # requested validation observation count
    mean_state::Float64 # conserved mean used by Cahn-Hilliard
end
# Appears in:
# - Core/types.jl
# - Core/pipeline.jl
# - Core/variable_windows.jl
# - Equations/allen_cahn.jl
# - Equations/cahn_hilliard.jl
# - Equations/reaction_diffusion.jl

struct ROMData
    state_modes::Matrix{Float64} # state POD basis
    nonlinear_modes::Matrix{Float64} # nonlinear POD/DEIM basis
    deim_indices::Vector{Int} # sampled nonlinear-state indices
    linear_operator::Matrix{Float64} # reduced linear operator
    sampled_state::Matrix{Float64} # first sampled state reconstruction map
    sampled_state_2::Union{Nothing,Matrix{Float64}} # second sampled state map for reaction-diffusion
    nonlinear_projection::Matrix{Float64} # DEIM map from sampled values to reduced RHS
    mean_state::Float64 # ROM centering value for Cahn-Hilliard
    components::Union{Nothing,Vector{Int}} # reaction-diffusion component for each DEIM point
    spatial_indices::Union{Nothing,Vector{Int}} # spatial index corresponding to each DEIM point
    state_singular_values::Vector{Float64} # state snapshot singular values
    nonlinear_singular_values::Vector{Float64} # nonlinear snapshot singular values
end
# Appears in:
# - Core/types.jl
# - Core/reduction.jl
# - Core/saving.jl
# - Equations/allen_cahn.jl
# - Equations/cahn_hilliard.jl
# - Equations/reaction_diffusion.jl

struct PreparedTraining{P,Q,R,J,K}
    config::RunConfig # immutable run configuration
    grid::Grid # spatial discretization
    learner::LearnerSetup # initialized learner
    reference::ReferenceData # reference trajectory and operator data
    mode::Symbol # FOM or ROM training mode
    problem::P # trainable ODE problem
    initial_parameters::Q # trainable ODE parameters before optimization
    rebuild_parameters::R # trainable-parameter reconstruction closure
    project::J # full-to-reduced initial-state projection or nothing
    reconstruct::K # reduced-to-full state reconstruction closure
    Δmeasure::Float64 # spatial quadrature weight
    rom::Union{Nothing,ROMData} # ROM data when mode is ROM
    equation_name::String # equation selector used by runners and metadata
end
# Appears in:
# - Core/types.jl
# - Core/pipeline.jl
# - Core/variable_windows.jl
# - Tools/Tests/runtests.jl
# - Core/saving.jl
# - Equations/allen_cahn.jl
# - Equations/cahn_hilliard.jl
# - Equations/reaction_diffusion.jl

struct TrainingConfig
    etas::Vector{Float64} # learning rate for each stage
    iterations::Vector{Int} # optimizer updates for each stage
    window_T::Vector{Float64} # physical window length for each stage
    window_N_obs::Vector{Int} # observations per training window for each stage
    window_start_policy::Vector{String} # beginning or random window policy for each stage
    loss_normalization::String # mean or sum trajectory-loss scaling
    loss_space::String # FULL or REDUCED ROM trajectory-loss comparison space
    window_seed::Int # deterministic random-window seed
    beta::Tuple{Float64,Float64} # Adam momentum coefficients
    warmup::Bool # whether to compile with one warmup update
    save_frequency::Int # parameter-history snapshot interval
    print_frequency::Int # progress-log interval
    learned_function_error::Bool # whether saved parameter snapshots include learned-function L2 error
    learned_function_error_bounds::Tuple{Float64,Float64} # shared lower and upper function-error integration bounds
end
# Appears in:
# - Core/types.jl
# - Core/cli.jl
# - Core/pipeline.jl
# - Core/saving.jl
# - Core/variable_windows.jl
# - Tools/Tests/runtests.jl

struct WindowSpec
    stage::Int # one-based learning-stage index
    iteration::Int # one-based update index within its stage
    policy::String # beginning, random, or validation policy
    t_start::Float64 # window start time
    t_end::Float64 # window end time
    window_T::Float64 # physical window duration
    n_obs::Int # comparison observations inside the window
    t_obs::Vector{Float64} # comparison times in the window
end
# Appears in:
# - Core/types.jl
# - Core/variable_windows.jl

# This tests whether two schedules are the same. Useful for testing only. 
Base.:(==)(left::WindowSpec, right::WindowSpec) =
    left.stage == right.stage && left.iteration == right.iteration && left.policy == right.policy &&
    left.t_start == right.t_start && left.t_end == right.t_end && left.window_T == right.window_T &&
    left.n_obs == right.n_obs && left.t_obs == right.t_obs

struct TrainingWindow
    spec::WindowSpec # immutable time-window specification
    u0::Vector{Float64} # full-order state at the window start
    model_u0::Vector{Float64} # FOM or ROM state used to start the model solve
    reference_observations::Vector{Vector{Float64}} # full-order comparison states
    model_reference_observations::Vector{Vector{Float64}} # reference states preprojected to the model comparison space
end
# Appears in:
# - Core/types.jl
# - Core/losses.jl
# - Core/reduction.jl
# - Core/variable_windows.jl

struct TrainingSnapshot
    iteration::Int # global optimizer update index
    stage::Int # completed learning stage
    kind::Symbol # :parameter for training or :validation for full-trajectory evaluation
    θ::Union{Nothing,Vector{Float64}} # trainable parameters for :parameter snapshots, otherwise nothing
    loss::Float64 # training-window or fixed validation loss according to kind
    learned_function_error::Union{Nothing,Float64} # optional learned-vs-true function L2 error for parameter snapshots
end
# Appears in:
# - Core/types.jl
# - Core/saving.jl
# - Core/variable_windows.jl
# - Tools/Tests/runtests.jl

struct TrainingOutput{R,T}
    result::R # optimizer result object
    final_theta::T # final flattened trainable parameters
    training::TrainingConfig # resolved training schedule used for the run
    parameter_history::Vector{TrainingSnapshot} # saved parameter snapshots
    final_training_loss::Float64 # final stochastic training-window loss
    final_full_trajectory_loss::Float64 # final full-trajectory validation loss
    window_history::Vector{WindowSpec} # all precomputed training windows
    validation_history::Vector{TrainingSnapshot} # initial and per-stage validation losses
end
# Appears in:
# - Core/types.jl
# - Core/pipeline.jl
# - Core/saving.jl
# - Core/variable_windows.jl
# - Tools/Tests/runtests.jl
