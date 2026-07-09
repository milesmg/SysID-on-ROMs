### ADJUSTED: Extend the existing FOM/ROM HPC helpers instead of duplicating their shared optimization code.

"""Evaluate `sum(coefficients[j+1] * u^j for j=0:degree)` by Horner's rule."""
function polynomial_value(u, coefficients)
    value = zero(u + first(coefficients))
    @inbounds for j in length(coefficients):-1:1
        value = value * u + coefficients[j]
    end
    return value
end

"""Evaluate the learned polynomial nonlinearity at every entry of `u`."""
polynomial_values(u, coefficients) = [polynomial_value(ui, coefficients) for ui in u]

"""Initial coefficient vector for a degree-`polynomial_degree` learned nonlinearity."""
function initial_polynomial_coefficients(polynomial_degree::Integer)
    polynomial_degree >= 0 || error("polynomial_degree must be nonnegative")
    return zeros(Float64, polynomial_degree + 1)
end

"""RHS of AC with a learned polynomial nonlinearity."""
function rhs_ac_polynomial!(du, u, p, t, dimension=1, N=length(u), boundary_condition="homogeneous_dirichlet")
    (; ε2, Δx, θ) = p
    lap_ac!(du, u, N, dimension, 1 / Δx^2, boundary_condition)
    f = polynomial_values(u, θ)
    @inbounds @simd for i in eachindex(du)
        du[i] = ε2 * du[i] + f[i]
    end
    return nothing
end

"""Set up an in-place polynomial FOM ODE problem."""
function polynomial_ODE_prob(u₀, tspan, p₀; N=nothing, dimension=1, boundary_condition="homogeneous_dirichlet")
    dim = validate_ac_dimension(dimension)
    bc = validate_ac_boundary_condition(boundary_condition)
    grid_N = isnothing(N) ? ac_grid_size(length(u₀), dim) : N
    u₀ = normalize_ac_initial_condition(u₀, grid_N, dim)
    rhs! = (du, u, p, t) -> rhs_ac_polynomial!(du, u, p, t, dim, grid_N, bc)
    jac_prototype = dim == 1 && bc == "homogeneous_dirichlet" ?
        Tridiagonal(
            zeros(eltype(u₀), grid_N - 1),
            zeros(eltype(u₀), grid_N),
            zeros(eltype(u₀), grid_N - 1),
        ) :
        get_lap_ac_matrix(grid_N, one(eltype(u₀)), one(eltype(u₀)), dim, bc)
    return ODEProblem(ODEFunction(rhs!; jac_prototype), u₀, tspan, p₀)
end

"""
Get the parameters for a polynomial FOM optimization.

This intentionally reuses the existing FOM loss, variable-window optimizer,
and save path; only the learned nonlinearity setup differs from the NN path.
"""
function prepare_for_optimization(;N=256,
                                L=1.0,
                                ε2 = 1e-2,
                                tspan = (0.0, 2.0),
                                N_obs = 10,
                                dimension = 1,
                                boundary_condition = "homogeneous_dirichlet",
                                u₀ = nothing,
                                h = 8,
                                seed = 1,
                                polynomial_degree = 3,
                                polynomial_initial_coefficients = nothing,
                                )
    dimension = validate_ac_dimension(dimension)
    boundary_condition = validate_ac_boundary_condition(boundary_condition)
    grid = ac_grid(N, L, boundary_condition)
    Δx = grid.Δx
    x = grid.x
    u₀ = isnothing(u₀) ?
        default_ac_initial_condition(N, L, ε2, dimension, boundary_condition) :
        normalize_ac_initial_condition(u₀, N, dimension)
    t_obs = collect(LinRange(tspan[1] + (tspan[2]-tspan[1])/(N_obs-1), tspan[2], N_obs-1))
    θ₀ = isnothing(polynomial_initial_coefficients) ?
        initial_polynomial_coefficients(polynomial_degree) :
        Float64.(collect(polynomial_initial_coefficients))
    polynomial_degree = length(θ₀) - 1
    p₀ = ComponentVector(ε2=ε2, Δx=Δx, Δmeasure=spatial_measure(Δx, dimension), θ=θ₀)

    run_params = (;
        N, L, ε2, Δx,
        dimension,
        boundary_condition,
        state_shape=dimension == 1 ? (N,) : (N, N),
        x=copy(x),
        y=dimension == 1 ? nothing : copy(x),
        tspan,
        N_obs,
        t_obs=copy(t_obs),
        u₀=copy(u₀),
        learner="polynomial",
        model_type="polynomial",
        polynomial_degree,
        polynomial_coefficient_order="ascending powers of u",
        polynomial_initial_coefficients=copy(θ₀),
        seed,
    )

    prob = polynomial_ODE_prob(u₀, tspan, p₀; N, dimension, boundary_condition)
    return (; prob, p₀, t_obs, x, u₀, run_params)
end

"""Evaluate the Allen-Cahn POD/DEIM polynomial ROM right-hand side."""
function rhs_ac_polynomial_ROM!(dũ, ũ, p, t)
    (; Ã, θ, Up, B) = p
    z = Up * ũ
    dũ .= Ã * ũ + B * polynomial_values(z, θ)
    return nothing
end

"""Construct the reduced polynomial `ODEProblem`."""
function polynomial_ROM_problem(ũ₀, tspan, p₀, data)
    f = ODEFunction((dũ, ũ, p, t) -> begin
        data.rom
        rhs_ac_polynomial_ROM!(dũ, ũ, p, t)
    end)
    return ODEProblem(f, ũ₀, tspan, p₀)
end

"""
Prepare an Allen-Cahn POD/DEIM polynomial ROM optimization.

This reuses the existing ROM builder, loss, variable-window optimizer, and
save path; only NN setup is replaced with polynomial coefficients.
"""
function prepare_ROM_optimization(
    A,
    u_ref,
    r,
    m;
    nonlinear_snapshots=nothing,
    Δx=u_ref.prob.p.Δx,
    N_obs=10,
    t_obs=collect(LinRange(
        u_ref.prob.tspan[1] + (u_ref.prob.tspan[2] - u_ref.prob.tspan[1]) / (N_obs - 1),
        u_ref.prob.tspan[2],
        N_obs - 1,
    )),
    h=8,
    seed=1,
    dimension=1,
    boundary_condition=hasproperty(u_ref.prob.p, :boundary_condition) ? u_ref.prob.p.boundary_condition : "homogeneous_dirichlet",
    polynomial_degree=3,
    polynomial_initial_coefficients=nothing,
)
    dimension = validate_ac_dimension(dimension)
    boundary_condition = validate_ac_boundary_condition(boundary_condition)
    u_snapshots = hcat(u_ref.u...)
    grid_N = ac_grid_size(size(u_snapshots, 1), dimension)
    k = u_ref.prob.p.k
    use_default_nonlinearity = isnothing(nonlinear_snapshots)
    fu_snapshots = use_default_nonlinearity ?
        .-k .* (u_snapshots .^ 3 .- u_snapshots) :
        (nonlinear_snapshots isa AbstractMatrix ? nonlinear_snapshots : hcat(nonlinear_snapshots...))
    rom = build_rom(A, u_snapshots, fu_snapshots, r, m)
    θ₀ = isnothing(polynomial_initial_coefficients) ?
        initial_polynomial_coefficients(polynomial_degree) :
        Float64.(collect(polynomial_initial_coefficients))
    polynomial_degree = length(θ₀) - 1

    data = (;
        rom,
        u_ref_obs=hcat((u_ref(ti) for ti in t_obs)...),
        t_obs=copy(t_obs),
        Δx,
        Δmeasure=spatial_measure(Δx, dimension),
        dimension,
        boundary_condition,
        state_shape=dimension == 1 ? (grid_N,) : (grid_N, grid_N),
        N=size(u_snapshots, 1),
        grid_N,
        ε2=u_ref.prob.p.ε2,
        k,
        full_u₀=copy(u_ref.prob.u0),
        reference_saved_times=copy(u_ref.t),
        reference_algorithm=string(nameof(typeof(u_ref.alg))),
        h=nothing,
        activation="polynomial",
        learner="polynomial",
        model_type="polynomial",
        polynomial_degree,
        polynomial_coefficient_order="ascending powers of u",
        polynomial_initial_coefficients=copy(θ₀),
        seed,
        requested_N_obs=N_obs,
        use_default_nonlinearity,
    )

    p₀ = ComponentVector(Ã=rom.Ã, Up=rom.Up, B=rom.B, θ=θ₀)
    ũ₀ = rom.U' * u_ref.prob.u0
    return polynomial_ROM_problem(ũ₀, u_ref.prob.tspan, p₀, data)
end
