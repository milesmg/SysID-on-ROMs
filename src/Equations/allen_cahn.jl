# Build allen-cahn-specific tooling 

"""
AC nonlinearity via a learned function
Args:
- u: state (can be select points)
- init: LearnerSetup struct
- θ: params
"""
ac_values(u, init::LearnerSetup, θ) = init.kind == "nn" ? nn_values(u, init.nn, θ, init.state) : polynomial_values(u, θ)

"""
AC cubic nonlinearity
Args:
- u: state (can be select points)
- k: coefficient (default 1.0)
"""
ac_fixed_values(u, k) = .-k .* (u .^ 3 .- u)

"""
AC initial condition defaults. 1 dim is the tanh activation; 2 dim is a circle-like thing. 
Args:
- grid: a Grid struct (see types.jl)
- ε2: coefficient
"""
function ac_default_state(grid::Grid, ε2)
    x = grid.x
    grid.dimension == 1 && return tanh.((x .- grid.L / 2) ./ sqrt(2ε2))
    vec([tanh((hypot(x[i] - grid.L / 2, x[j] - grid.L / 2) - grid.L / 4) / sqrt(2ε2)) for i in 1:grid.N, j in 1:grid.N])
end

"""
Builds a dict with named AC initial conditions, either in 1D or in 2D, then returns the one with the correct name
Args:
- name: the name of the initial condition
    - 1D: 'step', 'low frequency sine', 'high frequency sine', 'off center bump' 
    - 2D: '2d circle drop', '2d offcenter drop', '2d one direction tanh front', '2d annulus', '2d sin xy', '2d high frequency x sin', '2d random noise' 
- grid: a Grid struct (see types.jl)
- ε2: coefficient
"""
function ac_named_state(name, grid::Grid, ε2)
    x, L, w = grid.x, grid.L, sqrt(2ε2)
    if grid.dimension == 1
        f = Dict("step" => x -> x < L / 2 ? -1.0 : 1.0,
                 "low frequency sine" => x -> sin(2π*x/L),
                 "high frequency sine" => x -> sin(8π*x/L),
                 "off center bump" => x -> exp(-((x - .35L)^2) / (2(.08L)^2)))[name]
        return [f(xi) for xi in x]
    else
        f = Dict(
            "2d circle drop" => (x, y) -> tanh((hypot(x - .5L, y - .5L) - .23L) / w),
            "2d offcenter drop" => (x, y) -> tanh((hypot(x - .35L, y - .58L) - .18L) / w),
            "2d one direction tanh front" => (x, y) -> tanh((x - .5L) / w),
            "2d annulus" => (x, y) -> max(tanh((hypot(x - .5L, y - .5L) - .30L) / w), tanh((.15L - hypot(x - .5L, y - .5L)) / w)),
            "2d sin xy" => (x, y) -> sin(2π*x/L) * sin(2π*y/L),
            "2d high frequency x sine" => (x, y) -> sin(3π*x/L),
            "2d random noise" => (x, y) -> 0.1 * randn(),
        )[name]
        [f(x[i], x[j]) for i in 1:grid.N, j in 1:grid.N]
    end
end

"""
Runs the AC reference solution. Returns a ReferenceData struct. 
Args:
- config: a RunConfig struct
- grid: a Grid struct
- u₀: the initial condition
"""
function ac_reference(config::RunConfig, grid::Grid, u₀::Vector{Float64})::ReferenceData
    p = config.parameters
    function rhs!(du,u,p,t)
        laplacian!(du, u, grid)
        @. du = p.ε2 * du - p.k * (u^3 - u)
    end    
    Δt = config.reference_dt_factor * grid.Δx^2 / (2p.ε2)
    tspan = (0.0, config.tfinal)
    t = reference_save_times(config.tfinal)
    prob = ODEProblem(ODEFunction(rhs!; jac_prototype=laplacian_matrix(grid; scale=p.ε2)), u₀, tspan, p)
    ReferenceData(solve(prob, TRBDF2(); saveat=t), prob, p, copy(u₀), nothing, t, tspan, Δt, config.N_obs, 0.0)
end

"""
Builds ODE problems and their containing PreparedTraining structs for optimization, based on :fom vs :rom Symbol
Args:
- mode: :fom or :rom
- config: RunConfig struct
- grid: Grid struct
- reference: ReferenceData struct
- init: LearnerSetup struct
"""
function ac_model(mode::Symbol, config::RunConfig, grid::Grid, reference::ReferenceData, init::LearnerSetup)::PreparedTraining
    θ = init.θ
    p = config.parameters
    if mode == :fom
        p₀ = ComponentVector(ε2=p.ε2, Δx=grid.Δx, Δmeasure=spatial_measure(grid), θ=θ)
        function rhs!(du,u,p,t)
            laplacian!(du, u, grid)
            f = ac_values(u, init, p.θ)
            @. du = p.ε2 * du + f
        end  
        prob = ODEProblem(ODEFunction(rhs!; jac_prototype=laplacian_matrix(grid)), reference.initial_state, reference.tspan, p₀)
        rebuild = (p, re, θ) -> ComponentVector(ε2=p.ε2, Δx=p.Δx, Δmeasure=p.Δmeasure, θ=re(θ))
        return PreparedTraining(config, grid, init, reference, mode, prob, p₀, rebuild, nothing,
                                (state, _, _) -> state, spatial_measure(grid), nothing, "ac")
    elseif mode == :rom
        frames = hcat(reference.solution.u...)
        A = laplacian_matrix(grid; scale=p.ε2)
        U, state_singular_values = pod_modes(frames, p.r)
        V, nonlinear_singular_values = pod_modes(ac_fixed_values(frames, p.k), p.m)
        points = deim_indices(V)
        rom = ROMData(Matrix(U), Matrix(V), points, Matrix(U' * A * U), Matrix(U[points, :]), nothing,
                    Matrix(deim_projection(V, points, U)), 0.0, nothing, nothing,
                    Vector(state_singular_values), Vector(nonlinear_singular_values))
        p₀ = ComponentVector(Ã=rom.linear_operator, Up=rom.sampled_state, B=rom.nonlinear_projection, θ=θ)
        rhs! = (du, a, p, t) -> (du .= p.Ã * a + p.B * ac_values(p.Up * a, init, p.θ))
        prob = ODEProblem(ODEFunction(rhs!), U' * reference.initial_state, reference.tspan, p₀)
        rebuild = (p, re, θ) -> ComponentVector(Ã=p.Ã, Up=p.Up, B=p.B, θ=re(θ))
        PreparedTraining(config, grid, init, reference, mode, prob, p₀, rebuild, u -> rom.state_modes' * u,
                        (state, _, _) -> rom.state_modes * state, spatial_measure(grid), rom, "ac")
    end
end

"""
Build EquationSpec struct for AC
"""
function ac_spec()::EquationSpec
    EquationSpec("ac", 1, 1, 256, 2.0, 2, "dirichlet",
       options -> EquationParameters(; ε2=get_float(options, "eps2", 1e-5), k=get_float(options, "k", 1.0), r=get_int(options, "r", 20), m=get_int(options, "m", 10)),
       (grid, config) -> ac_default_state(grid, config.parameters.ε2),
       (name, grid, config) -> ac_named_state(name, grid, config.parameters.ε2),
       ac_reference, ac_model)
end
