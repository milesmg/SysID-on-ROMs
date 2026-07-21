# Build cahn-hilliard-specific tooling 

"""
CH nonlinearity via a learned function
Args:
- u: state (can be select points)
- init: LearnerSetup struct
- θ: params
"""
ch_values(u, init::LearnerSetup, θ) = init.kind == "nn" ? nn_values(u, init.nn, θ, init.state) : polynomial_values(u, θ)
"""
CH cubic nonlinearity
Args:
- u: state (can be select points)
- k: coefficient (default 1.0)
"""
ch_fixed_values(u) = u .^ 3 .- u

"""
CH initial condition defaults. 1 dim cosine wave with a mean_c, 2 dim is a checkerboard around a mean_c
Args:
- grid: a Grid struct (see types.jl)
- mean_c: float, the mean mass
"""
function ch_default_state(grid::Grid, mean_c)
    x = grid.x
    grid.dimension == 1 && return mean_c .+ .1 .* cos.(4π .* x ./ grid.L)
    vec([mean_c + .1cos(4π*x[i]/grid.L) * cos(4π*x[j]/grid.L) for i in 1:grid.N, j in 1:grid.N])
end


"""
CH named initial condition. The only one at the moment is "2d random scalar field" which requires dimension = 2
Args: 
- name: name of initial condition
- grid: a Grid struct
- config: a RunConfig struct
"""
function ch_named_state(name, grid::Grid, config::RunConfig)
    grid.dimension == 2 || error("2 dims required for random init")
    if name == "2d random scalar field"
        rng = MersenneTwister(config.seed)
        values = randn(rng, grid.N, grid.N)
        values .-= mean(values)
        return config.parameters.mean_c .+ .1 .* vec(values ./ maximum(abs.(values)))
    else 
        error("Unknown C-H initial condition name")
    end
end

"""
Define the Cahn-Hilliard ODE problem
Args:
- u₀: initial condition
- tspan: initial and final times
- p: an EquationParameters struct
- grid: a Grid struct
- D: the Laplacian matrix with some boundary condits
- nonlinearity: the nonlinearity in CH, either learned or fixed
"""
### ADJUSTED: Accept the trainable ComponentVector as well as fixed EquationParameters.
function ch_problem(u₀, tspan, p, grid::Grid, D, nonlinearity)
    Dc, Df = similar(u₀), similar(u₀)
    rhs! = (du, c, p, t) -> begin
        laplacian!(Dc, c, grid; scale=-one(eltype(c)))
        laplacian!(du, Dc, grid; scale=-one(eltype(c)))
        laplacian!(Df, nonlinearity(c, hasproperty(p, :θ) ? p.θ : nothing), grid; scale=-one(eltype(c)))
        @. du = -p.ε2 * du - Df - p.sigma * (c - p.mean_c)
    end
    ODEProblem(ODEFunction(rhs!; jac_prototype=D * D + D + sparse(I, size(D, 1), size(D, 1))), u₀, tspan, p)
end

"""
Build a CH reference object, stored as a ReferenceData struct. 
Args:
- config: a RunConfig struct
- grid: a Grid struct
- u₀: the initial condition vector
"""
function ch_reference(config::RunConfig, grid::Grid, u₀::Vector{Float64})::ReferenceData
    D = -laplacian_matrix(grid)
    mean_c = mean(u₀)
    p = EquationParameters(; ε2=config.parameters.ε2, sigma=config.parameters.sigma, mean_c, r=config.parameters.r, m=config.parameters.m)
    nonlinearity = (u, _) -> ch_fixed_values(u)
    Δt = config.reference_dt_factor * grid.Δx^4 / p.ε2
    tspan = (0.0, config.tfinal)
    t = collect(LinRange(0.0, config.tfinal, min(max(2, floor(Int, config.tfinal / Δt) + 1), 500)))
    prob = ch_problem(u₀, tspan, p, grid, D, nonlinearity)
    ReferenceData(solve(prob, TRBDF2(autodiff=AutoFiniteDiff()); saveat=t), prob, p, copy(u₀), D, t, tspan, Δt, config.N_obs, mean_c)
end


"""
Build a Cahn-Hilliard ROMData struct. I have added comments to the inverse laplacian computation. 
Args:
- D: (discrete) Laplacian matrix with relevant boundary condits
- frames: state snapshots from reference trajectory
- r: number of POD modes
- m: number of DEIM points / modes
- sigma: σ from C-H eqn
- mean_c: mean concentration fro C-H eqn
"""
function ch_rom(D, frames, r, m, ε2, sigma, mean_c)::ROMData
    # build a zero-mean spatial basis
    centered = frames .- mean(frames; dims=1)
    Phi, state_singular_values = pod_modes(centered, r)
    V, nonlinear_singular_values = pod_modes(ch_fixed_values(frames), m)
    points = deim_indices(V)
    n = size(Phi, 1)

    # here, we build the inverse laplacian necessary for our Petrov-Galerkin ROM. 
    # we're solving a 'constrained inverse' problem: We want to invert our periodic D, but it has nontrivial kernel: every flat vector is sent to 0.
    # thus, we augment our inverse problem: we want the matrix Ddag such that DDdag(v) = DagD(v) = 1, so long as mean(v)=0; that is, so long as 1^Tv = 0. 
    # Really, we want the {zₖ} such that Dzₖ = ϕₖ given 1ᵀzₖ=0, where the ϕₖ are our spatial basis. 
    # Since there's some numerical imprecision in zero-mean centering, we really need to do Dzₖ + λₖ1 = ϕₖ.
    # this computation is stored in Z. 
    augmented = [D sparse(ones(n, 1)); sparse(reshape(ones(n), 1, n)) spzeros(1, 1)]
    Z = (lu(augmented) \ vcat(Phi, zeros(1, size(Phi, 2))))[1:n, :]
    # here we build our P-G ROM 
    G = Matrix(Symmetric(Phi' * Z))
    K = Matrix(Symmetric(Phi' * D * Phi))
    Gfac = cholesky(Symmetric(G))
    B = (Phi' * V) / V[points, :]
    ROMData(Matrix(Phi), Matrix(V), points, Matrix(-(Gfac \ (ε2 * K + sigma * G))), Matrix(Phi[points, :]),
            nothing, Matrix(-(Gfac \ B)), mean_c, nothing, nothing,
            Vector(state_singular_values), Vector(nonlinear_singular_values))
end

"""
Build a trainable FOM or ROM of the Cahn-Hilliard equation as a PreparedTraining struct
Args:
- mode: :fom or :rom
- config: RunConfig struct
- grid: Grid struct
- reference: ReferenceData struct
- init: LearnerSetup struct
"""
function ch_model(mode::Symbol, config::RunConfig, grid::Grid, reference::ReferenceData, init::LearnerSetup)::PreparedTraining
    θ = init.θ
    p = config.parameters
    if mode == :fom
        p₀ = ComponentVector(ε2=p.ε2, sigma=p.sigma, mean_c=reference.mean_state, Δx=grid.Δx, Δmeasure=spatial_measure(grid), θ=θ)
        nonlinearity = (u, θ) -> ch_values(u, init, θ)
        prob = ch_problem(reference.initial_state, reference.tspan, p₀, grid, reference.operator, nonlinearity)
        rebuild = (p, re, θ) -> ComponentVector(ε2=p.ε2, sigma=p.sigma, mean_c=p.mean_c, Δx=p.Δx, Δmeasure=p.Δmeasure, θ=re(θ))
        return PreparedTraining(config, grid, init, reference, mode, prob, p₀, rebuild, nothing,
                                (state, _, _) -> state, spatial_measure(grid), nothing,"ch")
    elseif mode == :rom
        rom = ch_rom(reference.operator, hcat(reference.solution.u...), p.r, p.m, p.ε2, p.sigma, reference.mean_state)
        p₀ = ComponentVector(Atilde=rom.linear_operator, Phi_p=rom.sampled_state, Btilde=rom.nonlinear_projection, θ=θ)
        rhs! = (du, a, p, t) -> (du .= p.Atilde * a + p.Btilde * ch_values(rom.mean_state .+ p.Phi_p * a, init, p.θ))
        prob = ODEProblem(ODEFunction(rhs!), rom.state_modes' * (reference.initial_state .- rom.mean_state), reference.tspan, p₀)
        rebuild = (p, re, θ) -> ComponentVector(Atilde=p.Atilde, Phi_p=p.Phi_p, Btilde=p.Btilde, θ=re(θ))
        PreparedTraining(config, grid, init, reference, mode, prob, p₀, rebuild,
                     u -> rom.state_modes' * (u .- rom.mean_state),
                     (state, _, _) -> rom.mean_state .+ rom.state_modes * state, spatial_measure(grid), rom,"ch")

    end
end

"""
Build EquationSpec struct for CH
"""
function ch_spec()::EquationSpec
    EquationSpec("ch", 1, 1, 256, 10.0, 2, "periodic",
       options -> EquationParameters(; ε2=get_float(options, "eps2", 1e-2), sigma=get_float(options, "sigma", 1.0), mean_c=get_float(options, "mean-c", 0.0), r=get_int(options, "r", 20), m=get_int(options, "m", 10)),
       (grid, config) -> ch_default_state(grid, config.parameters.mean_c),
       ch_named_state, ch_reference, ch_model)
end
