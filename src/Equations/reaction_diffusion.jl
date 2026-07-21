# build reaction-diffusion-specific tooling
"""Default reaction diffusion first-component reaction function. 
Args:
- s1, s2 are the components"""
rd_s1(v1, v2) = v1 - v1^3 - v2 - .005
"""Default reaction diffusion second-component reaction function. 
Args:
- s1, s2 are the components"""
rd_s2(v1, v2) = 10 * (v1 - v2)

"""
Apply learned RHS nonlinearity for reaction diffusion
Args:
- v1: flattened vectorized first component
- v2: flattened vectorized second component
- init: LearnerSetup struct
- θ: learned function params
"""
rd_values(v1, v2, init::LearnerSetup, θ) = init.kind == "nn" ? nn_values(v1, v2, init.nn, θ, init.state) : rd_polynomial_values(v1, v2, θ)

"""
Apply learned RHS nonlinearity R-D, polynomial
Args:
- v1: flattened vectorized first component
- v2: flattened vectorized second component
- coefficients: polynomial coefficients
"""
function rd_polynomial_values(v1, v2, coefficients)
    degree = 0
    while (degree + 1) * (degree + 2) ÷ 2 < length(coefficients)
        degree += 1
    end
    exponents = [(i, total - i) for total in 0:degree for i in 0:total]
    [sum(coefficients[k] * v1[j]^i * v2[j]^l for (k, (i, l)) in enumerate(exponents)) for j in eachindex(v1)]
end

"""
Default reaction-diffusion initial condition
Args:
- grid: a Grid struct (see types.jl)
"""
function rd_default_state(grid::Grid)
    x, L = grid.x, grid.L
    if grid.dimension == 1
        v1 = 1 .+ .05 .* cos.(2π .* x ./ L)
        v2 = 1 .+ .05 .* sin.(2π .* x ./ L)
    else
        v1 = [1 + .05cos(2π*x[i]/L) * cos(2π*x[j]/L) for i in 1:grid.N, j in 1:grid.N]
        v2 = [1 + .05sin(2π*x[i]/L) * sin(2π*x[j]/L) for i in 1:grid.N, j in 1:grid.N]
    end
    vcat(vec(v1), vec(v2))
end

"""
Produce a named R-D initial condition.
     "random", the one I work with, is a random field with fluctuations of magnitude 0.05 centered at -³√(0.005)
Args:
- name: (str)
    - 1D: "sine", "random"
    - 2D: "two patches", "random"
- grid: a Grid struct
- config: a RunConfig struct
"""
function rd_named_state(name, grid::Grid, config::RunConfig)
    x, L, N = grid.x, grid.L, grid.N
    if name in ("sine",)
        return rd_default_state(grid)
    elseif name in ("two patches",) && grid.dimension == 2
        v1 = [1 + .15exp(-((x[i] - .30L)^2 + (x[j] - .50L)^2) / (.08L)^2) for i in 1:N, j in 1:N]
        v2 = [1 + .15exp(-((x[i] - .70L)^2 + (x[j] - .50L)^2) / (.08L)^2) for i in 1:N, j in 1:N]
        return vcat(vec(v1), vec(v2))
    elseif name == "random"
        rng = MersenneTwister(config.seed)
        n = N ^ grid.dimension
        base = -cbrt(.005)
        v1, v2 = base .+ .05 .* randn(rng, n), base .+ .05 .* randn(rng, n)
        return vcat(v1, v2)
    end
end

"""
Apply the RHS of the R-D eqn in-place
Args:
- du: the vector to be mutated
- u: current state
- p: params
- grid: a Grid struct
- n: length of single component in the concatenated state vector
- nonlinearity: the nonlinear part of the RHS
"""
function rd_rhs!(du, u, p, grid::Grid, n, nonlinearity)
    v1, v2 = @view(u[1:n]), @view(u[n+1:2n])
    Δv1, Δv2 = @view(du[1:n]), @view(du[n+1:2n])
    laplacian!(Δv1, v1, grid)
    laplacian!(Δv2, v2, grid)
    learned = nonlinearity(v1, v2, hasproperty(p, :θ) ? p.θ : nothing)
    @inbounds for i in 1:n
        du[i] = p.D1 * Δv1[i] + rd_s1(v1[i], v2[i])
        du[n+i] = p.D2 * Δv2[i] + learned[i]
    end
    nothing
end

"""
Build RD reference solution as a ReferenceData struct
Args:
- config: RunConfig struct
- grid: Grid struct
- u₀: initial condition
"""
function rd_reference(config::RunConfig, grid::Grid, u₀::Vector{Float64})::ReferenceData
    lap = laplacian_matrix(grid)
    n = spatial_length(grid)
    p = config.parameters
    Δt = config.reference_dt_factor * grid.Δx^2 / max(p.D1, p.D2)
    tspan = (0.0, config.tfinal)
    t = collect(LinRange(0.0, config.tfinal, min(max(2, floor(Int, config.tfinal / Δt) + 1), 500)))
    rhs! = (du, u, p, t) -> rd_rhs!(du, u, p, grid, n, (v1, v2, _) -> rd_s2.(v1, v2))
    jac = [p.D1 * lap sparse(I, n, n); sparse(I, n, n) p.D2 * lap]
    prob = ODEProblem(ODEFunction(rhs!; jac_prototype=jac), u₀, tspan, p)
    ReferenceData(solve(prob, TRBDF2(autodiff=AutoFiniteDiff()); saveat=t), prob, p, copy(u₀), lap, t, tspan, Δt, config.N_obs, 0.0)
end

"""
Build a RD ROM
Args: 
- lap: the discrete Laplacian matrix we're using
- frames: the state snapshots
- r: the number of spatial modes
- m: the number of DEIM points / modes
- D1: the diffusion coefficient on the first component
- D2: the diffusion coefficient on the second component
"""
function rd_rom(lap, frames, r, m, D1, D2)::ROMData
    Phi, state_singular_values = pod_modes(frames, r)
    n = size(frames, 1) ÷ 2
    F = similar(frames)
    for j in axes(frames, 2)
        v1, v2 = @view(frames[1:n, j]), @view(frames[n+1:2n, j])
        F[1:n, j] .= rd_s1.(v1, v2)
        F[n+1:2n, j] .= rd_s2.(v1, v2)
    end
    V, nonlinear_singular_values = pod_modes(F, m)
    points = deim_indices(V)
    spatial_points = [point <= n ? point : point - n for point in points]
    components = [point <= n ? 1 : 2 for point in points]
    L = [D1 * lap spzeros(n, n); spzeros(n, n) D2 * lap]
    ROMData(Matrix(Phi), Matrix(V), points, Matrix(Phi' * L * Phi), Matrix(Phi[spatial_points, :]),
            Matrix(Phi[n .+ spatial_points, :]), Matrix((Phi' * V) / V[points, :]), 0.0,
            components, spatial_points, Vector(state_singular_values), Vector(nonlinear_singular_values))
end


"""
Build a trainable FOM or ROM of the Reaction-Diffusion equation as a PreparedTraining struct
Args:
- mode: :fom or :rom
- config: RunConfig struct
- grid: Grid struct
- reference: ReferenceData struct
- init: LearnerSetup struct
"""
function rd_model(mode::Symbol, config::RunConfig, grid::Grid, reference::ReferenceData, init::LearnerSetup)::PreparedTraining
    θ = init.θ
    n = spatial_length(grid)
    p = config.parameters
    if mode == :fom
        p₀ = ComponentVector(D1=p.D1, D2=p.D2, Δx=grid.Δx, Δmeasure=spatial_measure(grid), θ=θ)
        rhs! = (du, u, p, t) -> rd_rhs!(du, u, p, grid, n, (v1, v2, θ) -> rd_values(v1, v2, init, θ))
        prob = ODEProblem(ODEFunction(rhs!), reference.initial_state, reference.tspan, p₀)
        rebuild = (p, re, θ) -> ComponentVector(D1=p.D1, D2=p.D2, Δx=p.Δx, Δmeasure=p.Δmeasure, θ=re(θ))
        return PreparedTraining(config, grid, init, reference, mode, prob, p₀, rebuild, nothing,
                                (state, _, _) -> state, spatial_measure(grid), nothing,"rd")
    elseif mode == :rom
        rom = rd_rom(reference.operator, hcat(reference.solution.u...), p.r, p.m, p.D1, p.D2)
        p₀ = ComponentVector(Atilde=rom.linear_operator, Phi_v1_p=rom.sampled_state, Phi_v2_p=rom.sampled_state_2, Btilde=rom.nonlinear_projection, θ=θ)
        rhs! = (du, a, p, t) -> begin
            v1, v2 = p.Phi_v1_p * a, p.Phi_v2_p * a
            sampled = [rom.components[j] == 1 ? rd_s1(v1[j], v2[j]) : rd_values(v1, v2, init, p.θ)[j] for j in eachindex(rom.components)]
            du .= p.Atilde * a + p.Btilde * sampled
        end
        prob = ODEProblem(ODEFunction(rhs!), rom.state_modes' * reference.initial_state, reference.tspan, p₀)
        rebuild = (p, re, θ) -> ComponentVector(Atilde=p.Atilde, Phi_v1_p=p.Phi_v1_p, Phi_v2_p=p.Phi_v2_p, Btilde=p.Btilde, θ=re(θ))
        PreparedTraining(config, grid, init, reference, mode, prob, p₀, rebuild, u -> rom.state_modes' * u,
                        (state, _, _) -> rom.state_modes * state, spatial_measure(grid), rom,"rd")

    end
end


"""
Build EquationSpec struct for RD
"""
function rd_spec()::EquationSpec
    EquationSpec("rd", 2, 2, 128, 32.0, 2, "neumann",
       options -> EquationParameters(; D1=get_float(options, "d1", 2.8e-4), D2=get_float(options, "d2", 5e-2), r=get_int(options, "r", 20), m=get_int(options, "m", 10)),
       (grid, config) -> rd_default_state(grid), rd_named_state, rd_reference, rd_model)
end
