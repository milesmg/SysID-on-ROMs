### ADJUSTED: Add POD-Galerkin DEIM RD ROM construction and trainable NN/polynomial s2 models.
if !isdefined(@__MODULE__, :rd_validate_dimension)
    include("FOM_opt_RD_hpc.jl")
end

##### RD POD And DEIM Construction #####

"""Build DEIM interpolation indices from nonlinear POD modes."""
function rd_deim_indices(V)
    m = size(V, 2)
    m > 0 || error("DEIM requires at least one nonlinear mode")
    points = Vector{Int}(undef, m)
    points[1] = argmax(abs.(V[:, 1]))
    for k in 2:m
        Vk = V[:, 1:k-1]
        pk = points[1:k-1]
        coeff = Vk[pk, :] \ V[pk, k]
        residual = V[:, k] - Vk * coeff
        points[k] = argmax(abs.(residual))
    end
    length(unique(points)) == m || error("DEIM selected duplicate interpolation indices")
    return points
end

"""Return raw-state POD modes for stacked `[v1; v2]` snapshots."""
function rd_state_pod(frames, r)
    r <= min(size(frames)...) || error("r=$r exceeds snapshot rank bound $(min(size(frames)...))")
    state_svd = svd(frames; full=false)
    return state_svd.U[:, 1:r], state_svd.S
end

"""Build stacked reference reaction snapshots from stacked RD states."""
function rd_reference_reaction_snapshots(frames)
    n2, n_snapshots = size(frames)
    iseven(n2) || error("stacked RD snapshots must have an even row count")
    n = n2 ÷ 2
    F = similar(frames)
    @inbounds for j in 1:n_snapshots
        v1 = @view frames[1:n, j]
        v2 = @view frames[n+1:2n, j]
        F[1:n, j] .= rd_s1_values(v1, v2)
        F[n+1:2n, j] .= rd_s2_true_values(v1, v2)
    end
    return F
end

"""Return nonlinear POD modes from reference or supplied reaction snapshots."""
function rd_nonlinear_pod(frames, m; nonlinear_snapshots=nothing)
    F = isnothing(nonlinear_snapshots) ? rd_reference_reaction_snapshots(frames) : (nonlinear_snapshots isa AbstractMatrix ? nonlinear_snapshots : hcat(nonlinear_snapshots...))
    m <= min(size(F)...) || error("m=$m exceeds nonlinear snapshot rank bound $(min(size(F)...))")
    nonlinear_svd = svd(F; full=false)
    return nonlinear_svd.U[:, 1:m], nonlinear_svd.S
end

"""Construct the full stacked RD diffusion operator."""
function rd_block_diffusion(lap, D1, D2)
    n = size(lap, 1)
    return [D1 .* lap spzeros(eltype(lap), n, n); spzeros(eltype(lap), n, n) D2 .* lap]
end

"""Build the offline POD-Galerkin DEIM operators and sampled basis data."""
function build_rd_pg_deim_rom(lap, frames, r, m; D1, D2, nonlinear_snapshots=nothing)
    size(lap, 1) == size(frames, 1) ÷ 2 || error("laplacian size must match one RD species")
    Phi, state_singular_values = rd_state_pod(frames, r)
    V, nonlinear_singular_values = rd_nonlinear_pod(frames, m; nonlinear_snapshots)
    points = rd_deim_indices(V)
    n = size(lap, 1)
    components = [point <= n ? 1 : 2 for point in points]
    spatial_points = [point <= n ? point : point - n for point in points]
    Lfull = rd_block_diffusion(lap, D1, D2)
    B = (Phi' * V) / V[points, :]
    return (; Phi, V, points, components, spatial_points, Phi_v1_p=Matrix(Phi[spatial_points, :]), Phi_v2_p=Matrix(Phi[n .+ spatial_points, :]), B, Atilde=Matrix(Phi' * Lfull * Phi), Btilde=Matrix(B), state_singular_values, nonlinear_singular_values, D1, D2)
end

##### RD ROM RHS And Setup #####

"""Evaluate sampled fixed or learned reaction values at a reduced state."""
function rd_sampled_reaction(a, p, rom, learner, nn, state, exponents)
    v1 = rom.Phi_v1_p * a
    v2 = rom.Phi_v2_p * a
    s2 = if learner == "nn"
        rd_nn_batch(v1, v2, nn, p.θ, state)
    elseif learner == "polynomial"
        rd_polynomial_values(v1, v2, p.θ, exponents)
    else
        rd_s2_true_values(v1, v2)
    end
    sampled = similar(v1)
    @inbounds for j in eachindex(sampled)
        sampled[j] = rom.components[j] == 1 ? rd_s1(v1[j], v2[j]) : s2[j]
    end
    return sampled
end

"""Evaluate the fixed reference RD ROM RHS."""
function rhs_rd_fixed_ROM!(da, a, p, t, data)
    da .= data.rom.Atilde * a + data.rom.Btilde * rd_sampled_reaction(a, p, data.rom, "fixed", nothing, nothing, data.exponents)
    return nothing
end

"""Evaluate the NN-learned RD ROM RHS."""
function rhs_rd_NN_ROM!(da, a, p, t, data)
    da .= data.rom.Atilde * a + data.rom.Btilde * rd_sampled_reaction(a, p, data.rom, "nn", data.nn, data.state, data.exponents)
    return nothing
end

"""Evaluate the polynomial-learned RD ROM RHS."""
function rhs_rd_polynomial_ROM!(da, a, p, t, data)
    da .= data.rom.Atilde * a + data.rom.Btilde * rd_sampled_reaction(a, p, data.rom, "polynomial", nothing, nothing, data.exponents)
    return nothing
end

"""Construct a reduced RD ODE problem for fixed, NN, or polynomial `s2`."""
function rd_ROM_problem(a₀, tspan, p₀, data)
    rhs! = data.learner == "fixed" ? ((da, a, p, t) -> rhs_rd_fixed_ROM!(da, a, p, t, data)) : data.learner == "nn" ? ((da, a, p, t) -> rhs_rd_NN_ROM!(da, a, p, t, data)) : ((da, a, p, t) -> rhs_rd_polynomial_ROM!(da, a, p, t, data))
    return ODEProblem(ODEFunction(rhs!), a₀, tspan, p₀)
end

"""Build a fixed true-reaction RD ROM for stability checks."""
function build_RD_ROM_reference(reference, r, m; nonlinear_snapshots=nothing)
    u_ref = reference.u_ref
    p_ref = reference.prob.p
    frames = hcat(u_ref.u...)
    rom = build_rd_pg_deim_rom(reference.lap, frames, r, m; D1=p_ref.D1, D2=p_ref.D2, nonlinear_snapshots)
    data = (; rom, learner="fixed", nn=nothing, state=nothing, exponents=rd_polynomial_exponents(3), N=p_ref.N, dimension=p_ref.dimension, state_shape=u_ref.prob.p.dimension == 1 ? (2, p_ref.N) : (2, p_ref.N, p_ref.N), boundary_condition="neumann", reference_algorithm=string(nameof(typeof(u_ref.alg))))
    p₀ = (; Atilde=rom.Atilde, Phi_v1_p=rom.Phi_v1_p, Phi_v2_p=rom.Phi_v2_p, Btilde=rom.Btilde)
    a₀ = rom.Phi' * u_ref.prob.u0
    return rd_ROM_problem(a₀, u_ref.prob.tspan, p₀, data)
end

"""Prepare a trainable POD-Galerkin DEIM RD ROM."""
function prepare_RD_ROM_optimization(reference, r, m; learner="nn", h=8, seed=1, polynomial_degree=3, polynomial_initial_coefficients=nothing, nonlinear_snapshots=nothing, N_obs=10)
    learner = rd_validate_learner(learner)
    learner in ("nn", "polynomial") || error("trainable RD ROM learner must be nn or polynomial")
    u_ref = reference.u_ref
    p_ref = reference.prob.p
    frames = hcat(u_ref.u...)
    rom = build_rd_pg_deim_rom(reference.lap, frames, r, m; D1=p_ref.D1, D2=p_ref.D2, nonlinear_snapshots)
    nn = nothing
    state = nothing
    exponents = rd_polynomial_exponents(polynomial_degree)
    θ₀ = if learner == "nn"
        rng = MersenneTwister(seed)
        nn = Chain(Dense(2 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
        ps₀, state₀ = Lux.setup(rng, nn)
        state = state₀
        fmap(x -> Float64.(x), ps₀)
    else
        isnothing(polynomial_initial_coefficients) ? rd_initial_polynomial_coefficients(polynomial_degree) : Float64.(collect(polynomial_initial_coefficients))
    end
    N = p_ref.N
    dimension = p_ref.dimension
    Δmeasure = p_ref.Δmeasure
    ### ADJUSTED: Match AC/CH observation timing so N_obs=2 produces the final-time sample.
    t_obs = collect(LinRange(u_ref.prob.tspan[1] + (u_ref.prob.tspan[2] - u_ref.prob.tspan[1]) / (N_obs - 1), u_ref.prob.tspan[2], max(1, N_obs - 1)))
    data = (; rom, nn, state, exponents, learner, model_type=learner, N, grid_N=N, dimension, boundary_condition="neumann", state_shape=dimension == 1 ? (2, N) : (2, N, N), D1=p_ref.D1, D2=p_ref.D2, ε2=nothing, k=nothing, sigma=nothing, mean_c=nothing, Δx=p_ref.Δx, Δmeasure, t_obs=copy(t_obs), full_u₀=copy(u_ref.prob.u0), reference_saved_times=copy(u_ref.t), reference_algorithm=string(nameof(typeof(u_ref.alg))), state_components=("v1", "v2"), learned_component="s2", reference_reactions="s1=v1-v1^3-v2-0.005; s2=10*(v1-v2)", h=learner == "nn" ? h : nothing, activation=learner == "nn" ? "tanh" : "polynomial", polynomial_degree=learner == "polynomial" ? polynomial_degree : nothing, polynomial_coefficient_order=learner == "polynomial" ? "ascending total-degree monomials (i,j)" : nothing, polynomial_initial_coefficients=learner == "polynomial" ? copy(θ₀) : nothing, seed, requested_N_obs=N_obs, use_default_nonlinearity=isnothing(nonlinear_snapshots))
    p₀ = ComponentVector(Atilde=rom.Atilde, Phi_v1_p=rom.Phi_v1_p, Phi_v2_p=rom.Phi_v2_p, Btilde=rom.Btilde, θ=θ₀)
    a₀ = rom.Phi' * u_ref.prob.u0
    return rd_ROM_problem(a₀, u_ref.prob.tspan, p₀, data)
end

##### RD ROM Optimization #####

"""Materialize RD ROM window references and reduced initial states."""
function materialize_RD_ROM_batch(u_ref, rom, specs)
    return [merge(window, (; u0_rom=rom.Phi' * window.u0)) for window in materialize_batch(u_ref, specs)]
end

"""Compute one reconstructed spatially weighted RD ROM window loss."""
function variable_window_RD_ROM_loss(window, prob, p, alg, sensalg, normalization)
    data = prob.f.f.data
    window_prob = remake(prob; u0=window.u0_rom, tspan=(window.t_start, window.t_end), p=p)
    sol = solve(window_prob, alg; saveat=window.t_obs, sensealg=sensalg)
    total = zero(eltype(first(sol.u)))
    @inbounds for j in eachindex(sol.u, window.u_ref_obs)
        u_model = data.rom.Phi * sol.u[j]
        u_ref = window.u_ref_obs[j]
        @simd for i in eachindex(u_model, u_ref)
            total += abs2(u_model[i] - u_ref[i])
        end
    end
    loss = 0.5 * data.Δmeasure * total
    return normalization == "mean" ? loss / length(window.u_ref_obs) : loss
end

"""Average or sum RD ROM window losses."""
function variable_window_RD_ROM_batch_loss(batch, prob, p, alg, sensalg, normalization)
    total = zero(eltype(first(batch).u0_rom))
    for window in batch
        total += variable_window_RD_ROM_loss(window, prob, p, alg, sensalg, normalization)
    end
    return normalization == "mean" ? total / length(batch) : total
end

"""Run staged Adam training for an RD POD-Galerkin DEIM ROM."""
function run_variable_window_RD_ROM_optimization(u_ref, prob; eta=5e-2, beta=(0.9, 0.99), N_iter=400, window_T=nothing, window_N_obs=nothing, window_start_policy="beginning", batch_size=1, loss_normalization="mean", window_seed=1, validation_N_obs=prob.f.f.data.requested_N_obs, alg=TRBDF2(autodiff=AutoFiniteDiff()), sensalg=GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP()), warmup=true, save_frequency=nothing, print_frequency=10)
    data = prob.f.f.data
    core = run_variable_window_stages(u_ref, prob, prob.p; optimization_data=(; rom_prob=prob), materialize_model_batch=(reference, specs) -> materialize_RD_ROM_batch(reference, data.rom, specs), rebuild_params=(p, re, theta) -> ComponentVector(Atilde=p.Atilde, Phi_v1_p=p.Phi_v1_p, Phi_v2_p=p.Phi_v2_p, Btilde=p.Btilde, θ=re(theta)), batch_loss=variable_window_RD_ROM_batch_loss, validation_N_obs, log_name="run_variable_window_RD_ROM_optimization", eta, beta, N_iter, window_T, window_N_obs, window_start_policy, batch_size, loss_normalization, window_seed, alg, sensalg, warmup, save_frequency, print_frequency)
    return (; core.result, core.parameter_history, core.final_loss, core.final_training_loss, core.final_full_trajectory_loss, rom_prob=prob, run_settings=core.settings, core.window_history, core.validation_history)
end

##### RD ROM Save Helpers #####

### ADJUSTED: Save RD ROM data with the AC/CH raw-file layout and two-field DEIM metadata.
"""Save RD ROM optimization data under the standard run directory."""
function save_RD_ROM_optimization_data(output, run_name::AbstractString)
    data_root = normpath(joinpath(@__DIR__, "..", "..", "..", "Optimization", "Data"))
    run_directory = assert_run_name_available(run_name; data_root)
    mkpath(data_root)
    mkdir(run_directory)
    prob = output.rom_prob
    data = prob.f.f.data
    rom = data.rom
    polynomial_final_coefficients = data.model_type == "polynomial" ? copy(last(output.parameter_history).θ) : nothing
    rom_data = (; N=data.N, grid_N=data.grid_N, dimension=data.dimension, boundary_condition=data.boundary_condition, state_shape=data.state_shape, ε2=data.ε2, k=data.k, sigma=data.sigma, mean_c=data.mean_c, D1=data.D1, D2=data.D2, Δx=data.Δx, Δmeasure=data.Δmeasure, tspan=prob.tspan, t_obs=data.t_obs, u₀=data.full_u₀, reference_saved_times=data.reference_saved_times, r=size(rom.Phi, 2), m=size(rom.V, 2), spatial_modes=rom.Phi, deim_modes=rom.V, deim_indices=rom.points, deim_components=rom.components, deim_spatial_indices=rom.spatial_points, state_singular_values=rom.state_singular_values, nonlinear_singular_values=rom.nonlinear_singular_values, learner=data.learner, model_type=data.model_type, state_components=data.state_components, learned_component=data.learned_component, reference_reactions=data.reference_reactions, polynomial_degree=data.polynomial_degree, polynomial_coefficient_order=data.polynomial_coefficient_order, polynomial_initial_coefficients=data.polynomial_initial_coefficients, polynomial_final_coefficients, h=data.h, activation=data.activation, seed=data.seed, use_default_nonlinearity=data.use_default_nonlinearity)
    serialize(joinpath(run_directory, "parameter_history.jls"), output.parameter_history)
    serialize(joinpath(run_directory, "rom_data.jls"), rom_data)
    metadata = (; saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), julia_version=VERSION, equation="rd", N=rom_data.N, grid_N=rom_data.grid_N, dimension=rom_data.dimension, boundary_condition=rom_data.boundary_condition, state_shape=rom_data.state_shape, ε2=rom_data.ε2, k=rom_data.k, sigma=rom_data.sigma, mean_c=rom_data.mean_c, D1=rom_data.D1, D2=rom_data.D2, Δx=rom_data.Δx, Δmeasure=rom_data.Δmeasure, tspan=rom_data.tspan, N_obs=length(rom_data.t_obs), reference_steps=length(rom_data.reference_saved_times) - 1, r=rom_data.r, m=rom_data.m, learner=rom_data.learner, model_type=rom_data.model_type, state_components=rom_data.state_components, learned_component=rom_data.learned_component, reference_reactions=rom_data.reference_reactions, polynomial_degree=rom_data.polynomial_degree, polynomial_final_coefficients=rom_data.polynomial_final_coefficients, h=rom_data.h, activation=rom_data.activation, seed=rom_data.seed, reference_algorithm=data.reference_algorithm, output.run_settings..., initial_loss=first(output.parameter_history).loss, final_loss=output.final_loss, parameter_snapshots=length(output.parameter_history))
    open(joinpath(run_directory, "metadata.txt"), "w") do io
        for (name, value) in pairs(metadata)
            print(io, name, " = ")
            show(io, value)
            println(io)
        end
    end
    return run_directory
end

### ADJUSTED: Save RD variable-window ROM histories under the standard ROM filenames.
function save_variable_window_RD_ROM_optimization_data(output, run_name::AbstractString)
    run_directory = save_RD_ROM_optimization_data(output, run_name)
    serialize(joinpath(run_directory, "window_history.jls"), output.window_history)
    serialize(joinpath(run_directory, "validation_history.jls"), output.validation_history)
    serialize(joinpath(run_directory, "evaluation_history.jls"), output.validation_history)
    return run_directory
end
