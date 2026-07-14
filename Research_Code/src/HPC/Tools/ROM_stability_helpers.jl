### ADJUSTED: Centralize local ROM stability notebook helpers so notebooks only call shared tooling.
if !isdefined(@__MODULE__, :build_ac_reference)
    include(joinpath(@__DIR__, "hpc_common.jl"))
end

if !isdefined(@__MODULE__, :prepare_ROM_optimization)
    include(joinpath(@__DIR__, "..", "Simulations", "ROM_opt_AC_hpc.jl"))
end

if !isdefined(@__MODULE__, :prepare_CH_ROM_optimization)
    include(joinpath(@__DIR__, "..", "Simulations", "ROM_opt_CH_hpc.jl"))
end

"""Materialize an Allen-Cahn stability-test initial condition on the configured grid."""
function ac_materialize_stability_initial_condition(u0; N=256, L=1.0, ε2=1e-2, dimension=1, boundary_condition="homogeneous_dirichlet")
    dim = validate_ac_dimension(dimension)
    if isnothing(u0) || !(u0 isa Function)
        return u0
    end
    grid = ac_grid(N, L, boundary_condition)
    x = grid.x
    return dim == 1 ? Float64[u0(xi) for xi in x] : Float64[u0(x[i], x[j]) for i in 1:N, j in 1:N]
end

"""Plot Allen-Cahn stability-test initial conditions."""
function ac_plot_stability_initial_conditions(initial_conditions; N=256, L=1.0, ε2=1e-2, dimension=1, boundary_condition="homogeneous_dirichlet", show_colorbar=false)
    grid = ac_grid(N, L, boundary_condition)
    dim = validate_ac_dimension(dimension)
    if dim == 1
        p = plot(xlabel="x", ylabel="u0", title="Initial conditions")
        for item in initial_conditions
            values = isnothing(item.u0) ? default_ac_initial_condition(N, L, ε2, dim, boundary_condition) : ac_materialize_stability_initial_condition(item.u0; N, L, ε2, dimension=dim, boundary_condition)
            plot!(p, grid.x, values; label=item.name)
        end
        return p
    end
    plots = Any[]
    for item in initial_conditions
        values = isnothing(item.u0) ? default_ac_initial_condition(N, L, ε2, dim, boundary_condition) : ac_materialize_stability_initial_condition(item.u0; N, L, ε2, dimension=dim, boundary_condition)
        push!(plots, heatmap(grid.x, grid.x, reshape(values, N, N); title=item.name, aspect_ratio=:equal, clims=show_colorbar ? (-1, 1) : nothing, colorbar=show_colorbar))
    end
    return plot(plots...; layout=(1, length(plots)), size=(350 * length(plots), 320))
end

"""Run an Allen-Cahn FOM reference for local ROM stability checks."""
function ac_run_stability_fom_reference(; N=256, L=1.0, ε2=1e-2, k=1.0, tspan=(0.0, 2.0), reference_dt_factor=0.5, u0=nothing, dimension=1, boundary_condition="homogeneous_dirichlet")
    tspan[1] == 0.0 || error("build_ac_reference expects tspan to start at 0.0")
    u₀ = ac_materialize_stability_initial_condition(u0; N, L, ε2, dimension, boundary_condition)
    ref = build_ac_reference(; N, L, ε2, k, tfinal=tspan[2], reference_dt_factor, dimension, boundary_condition, u₀)
    A = get_lap_ac_matrix(N, ε2, ref.Δx, ref.dimension, ref.boundary_condition)
    return merge(ref, (; sol=ref.u_ref, A))
end

"""Build an Allen-Cahn ROM bundle for stability checks."""
function ac_build_stability_rom(fom, r::Integer, m::Integer; N_obs=100, h=8, seed=1)
    prob = prepare_ROM_optimization(fom.A, fom.u_ref, r, m; N_obs, h, seed, dimension=fom.dimension, boundary_condition=fom.boundary_condition)
    return (; prob, rom=prob.f.f.data.rom)
end

function ac_rhs_true_rom!(du, u, p, t)
    z = p.rom.Up * u
    fz = .-p.k .* (z .^ 3 .- z)
    du .= p.rom.Ã * u + p.rom.B * fz
    return nothing
end

"""Run a fixed-cubic Allen-Cahn ROM and reconstruct its full-state trajectory."""
function ac_run_stability_rom(fom, rom_bundle; alg=TRBDF2())
    rom = rom_bundle.rom
    u0_rom = rom.U' * fom.u_ref.prob.u0
    prob = ODEProblem(ac_rhs_true_rom!, u0_rom, fom.u_ref.prob.tspan, (; rom, k=fom.u_ref.prob.p.k))
    sol = solve(prob, alg; saveat=fom.u_ref.t)
    reconstructed = rom.U * hcat(sol.u...)
    return (; sol, reconstructed)
end

"""Materialize a Cahn-Hilliard stability-test initial condition on the configured grid."""
function ch_materialize_stability_initial_condition(u0; N=256, L=1.0, ε2=1e-2, dimension=1, mean_c=0.0, boundary_condition="periodic")
    dim = ch_validate_dimension(dimension)
    if isnothing(u0) || !(u0 isa Function)
        return u0
    end
    grid = ch_periodic_grid(N, L)
    x = grid.x
    return dim == 1 ? Float64[u0(xi) for xi in x] : Float64[u0(x[i], x[j]) for i in 1:N, j in 1:N]
end

"""Plot Cahn-Hilliard stability-test initial conditions."""
function ch_plot_stability_initial_conditions(initial_conditions; N=256, L=1.0, ε2=1e-2, dimension=1, mean_c=0.0, boundary_condition="periodic", show_colorbar=false)
    grid = ch_periodic_grid(N, L)
    dim = ch_validate_dimension(dimension)
    if dim == 1
        p = plot(xlabel="x", ylabel="c0", title="Initial conditions")
        for item in initial_conditions
            values = isnothing(item.u0) ? ch_default_initial_condition(N, L, dim; mean_c) : ch_materialize_stability_initial_condition(item.u0; N, L, ε2, dimension=dim, mean_c, boundary_condition)
            plot!(p, grid.x, values; label=item.name)
        end
        return p
    end
    plots = Any[]
    for item in initial_conditions
        values = isnothing(item.u0) ? ch_default_initial_condition(N, L, dim; mean_c) : ch_materialize_stability_initial_condition(item.u0; N, L, ε2, dimension=dim, mean_c, boundary_condition)
        push!(plots, heatmap(grid.x, grid.x, reshape(values, N, N); title=item.name, aspect_ratio=:equal, colorbar=show_colorbar))
    end
    return plot(plots...; layout=(1, length(plots)), size=(350 * length(plots), 320))
end

"""Run a Cahn-Hilliard FOM reference for local ROM stability checks."""
function ch_run_stability_fom_reference(; N=256, L=1.0, ε2=1e-2, sigma=1.0, mean_c=0.0, tspan=(0.0, 2.0), reference_dt_factor=0.5, u0=nothing, dimension=1, boundary_condition="periodic")
    u₀ = ch_materialize_stability_initial_condition(u0; N, L, ε2, dimension, mean_c, boundary_condition)
    ref = build_ch_reference(; N, L, ε2, sigma, tfinal=tspan[2], reference_dt_factor, dimension, u₀, mean_c)
    return merge(ref, (; sol=ref.u_ref, boundary_condition))
end

ch_normalize_stability_rom(rom) = merge(rom, (; U=rom.Phi, deim_indices=rom.points))

"""Build a Cahn-Hilliard ROM bundle for stability checks."""
function ch_build_stability_rom(fom, r::Integer, m::Integer; N_obs=100, h=8, seed=1)
    prob = prepare_CH_ROM_optimization(fom.D, fom.u_ref, r, m; N_obs, h, seed, dimension=fom.dimension, ε2=fom.ε2, sigma=fom.sigma, mean_c=fom.mean_c, Δx=fom.Δx, Δmeasure=fom.p₀.Δmeasure)
    return (; prob, rom=ch_normalize_stability_rom(prob.f.f.data.rom))
end

function ch_rhs_true_rom!(du, a, p, t)
    z = p.rom.mean_c .+ p.rom.Phi_p * a
    fz = ch_cubic_values(z)
    du .= p.rom.Atilde * a + p.rom.Btilde * fz
    return nothing
end

"""Run a fixed-cubic Cahn-Hilliard ROM and reconstruct its full-state trajectory."""
function ch_run_stability_rom(fom, rom_bundle; alg=TRBDF2(autodiff=AutoFiniteDiff()))
    rom = rom_bundle.rom
    a0 = rom.Phi' * (fom.u_ref.prob.u0 .- rom.mean_c)
    prob = ODEProblem(ch_rhs_true_rom!, a0, fom.u_ref.prob.tspan, (; rom))
    sol = solve(prob, alg; saveat=fom.u_ref.t)
    reconstructed = rom.mean_c .+ rom.Phi * hcat(sol.u...)
    return (; sol, reconstructed)
end

stability_capture_ratio(s, n) = sum(abs2, s[1:min(n, length(s))]) / sum(abs2, s)

"""Return state/nonlinear singular-value capture rows for a ROM builder."""
function stability_capture_table(fom, build_rom; rs=[2, 4, 8, 10, 15, 20], ms=[2, 4, 8, 10, 15, 20])
    max_r = minimum((maximum(rs), length(fom.u_ref.u), length(fom.u_ref.u[1])))
    max_m = minimum((maximum(ms), length(fom.u_ref.u), length(fom.u_ref.u[1])))
    base = build_rom(fom, max_r, max_m).rom
    rows = NamedTuple[]
    for r_i in rs, m_i in ms
        push!(rows, (;
            m=m_i,
            r=r_i,
            state_capture=stability_capture_ratio(base.state_singular_values, r_i),
            nonlinear_capture=stability_capture_ratio(base.nonlinear_singular_values, m_i),
        ))
    end
    return rows
end
