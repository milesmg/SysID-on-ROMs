if !isdefined(@__MODULE__, :build_ch_reference)
    include("FOM_opt_CH_hpc.jl")
end

if !isdefined(@__MODULE__, :run_variable_window_stages)
    include("variable_window_common_hpc.jl")
end


##### CH Petrov-Galerkin DEIM ROM Construction #####

"""Build DEIM interpolation indices from nonlinear POD modes."""
function ch_deim_indices(V)
    m = size(V, 2)
    points = Vector{Int}(undef, m)
    points[1] = argmax(abs.(V[:, 1]))
    for k in 2:m
        Vk = V[:, 1:k-1]
        pk = points[1:k-1]
        coeff = Vk[pk, :] \ V[pk, k]
        residual = V[:, k] - Vk * coeff
        points[k] = argmax(abs.(residual))
    end
    return points
end

"""Assert that every POD basis vector is zero mean within tolerance."""
function ch_assert_zero_mean_basis(Phi; tol=1e-8)
    column_means = vec(mean(Phi; dims=1))
    max_mean = maximum(abs.(column_means))
    max_mean <= tol || error("CH POD basis is not zero mean: max abs column mean = $max_mean, tol = $tol")
    return nothing
end

"""Compute `Ddagger * Phi` by sparse zero-mean constrained solves."""
function ch_Ddagger_basis(D, Phi)
    n, r = size(Phi)
    size(D, 1) == n && size(D, 2) == n || error("D must be $n x $n; got $(size(D))")
    ones_col = sparse(ones(eltype(Phi), n, 1))
    ones_row = sparse(reshape(ones(eltype(Phi), n), 1, n))
    augmented = [D ones_col; ones_row spzeros(eltype(Phi), 1, 1)]
    factor = lu(augmented)
    rhs = zeros(eltype(Phi), n + 1)
    Z = similar(Phi)
    for j in 1:r
        rhs[1:n] .= Phi[:, j]
        rhs[end] = zero(eltype(Phi))
        sol = factor \ rhs
        Z[:, j] .= sol[1:n]
    end
    return Z
end

"""Return POD state modes for zero-mean CH snapshots."""
function ch_state_pod(frames, r, #mean_c
            )
    r <= min(size(frames)...) || error("r=$r exceeds snapshot rank bound $(min(size(frames)...))")
    X = frames .- mean(frames; dims=1)
    state_svd = svd(X; full=false)
    # rank_tol = maximum(size(X)) * eps(eltype(X)) * maximum(state_svd.S)
    # effective_rank = count(>(rank_tol), state_svd.S)
    # r <= effective_rank || error("r=$r exceeds numerical zero-mean snapshot rank $effective_rank; largest discarded/requested singular value is $(state_svd.S[min(r, end)])")
    Phi = state_svd.U[:, 1:r]
    ch_assert_zero_mean_basis(Phi)
    return Phi, state_svd.S
end

"""Return nonlinear POD modes from full CH nonlinear snapshots or supplied snapshots."""
function ch_nonlinear_pod(frames, m; nonlinear_snapshots=nothing)
    F = if isnothing(nonlinear_snapshots)
        ### ADJUSTED: Match AC ROM behavior by building default nonlinear snapshots from full FOM states.
        ch_cubic_values(frames)
    else
        nonlinear_snapshots isa AbstractMatrix ? nonlinear_snapshots : hcat(nonlinear_snapshots...)
    end
    m <= min(size(F)...) || error("m=$m exceeds nonlinear snapshot rank bound $(min(size(F)...))")
    nonlinear_svd = svd(F; full=false)
    return nonlinear_svd.U[:, 1:m], nonlinear_svd.S
end

"""
Build the zero-mean H-minus-one Petrov-Galerkin DEIM ROM for nonlocal CH.

All inverse-Laplacian and reduced linear solves happen here, outside AD.
"""
function build_ch_pg_deim_rom(D, frames, r, m; ε2, sigma, mean_c, nonlinear_snapshots=nothing)
    size(D, 1) == size(frames, 1) && size(D, 2) == size(frames, 1) ||
        error("D must match the snapshot row count; got D=$(size(D)), frames=$(size(frames))")
    Phi, state_singular_values = ch_state_pod(frames, r)
    V, nonlinear_singular_values = ch_nonlinear_pod(frames, m; nonlinear_snapshots)
    points = ch_deim_indices(V)

    Z = ch_Ddagger_basis(D, Phi)
    G = Matrix(Symmetric(Phi' * Z))
    K = Matrix(Symmetric(Phi' * (D * Phi)))
    Gfac = cholesky(Symmetric(G))
    Phi_p = Matrix(Phi[points, :])
    B = (Phi' * V) / V[points, :]
    Atilde = -(Gfac \ (ε2 * K + sigma * G))
    Btilde = -(Gfac \ B)

    return (;
        Phi,
        V,
        points,
        Phi_p,
        B,
        G,
        K,
        Atilde=Matrix(Atilde),
        Btilde=Matrix(Btilde),
        mean_c,
        state_singular_values,
        nonlinear_singular_values,
    )
end


##### CH ROM RHS And Problem Setup #####

"""Evaluate the NN-learned CH Petrov-Galerkin DEIM ROM RHS."""
function rhs_ch_NN_ROM!(da, a, p, t, nn, state, mean_c)
    z = mean_c .+ p.Phi_p * a
    fz = ch_nn_batch(z, nn, p.θ, state)
    da .= p.Atilde * a + p.Btilde * fz
    return nothing
end

"""Evaluate the polynomial-learned CH Petrov-Galerkin DEIM ROM RHS."""
function rhs_ch_polynomial_ROM!(da, a, p, t, mean_c)
    z = mean_c .+ p.Phi_p * a
    fz = ch_polynomial_values(z, p.θ)
    da .= p.Atilde * a + p.Btilde * fz
    return nothing
end

"""Construct a CH reduced ODEProblem for NN or polynomial sampled chemistry."""
function ch_ROM_problem(a₀, tspan, p₀, data)
    f = if data.learner == "nn"
        ODEFunction((da, a, p, t) -> rhs_ch_NN_ROM!(da, a, p, t, data.nn, data.state, data.mean_c))
    else
        ODEFunction((da, a, p, t) -> rhs_ch_polynomial_ROM!(da, a, p, t, data.mean_c))
    end
    return ODEProblem(f, a₀, tspan, p₀)
end

"""
Prepare a trainable nonlocal CH Petrov-Galerkin DEIM ROM optimization.
"""
function prepare_CH_ROM_optimization(
    D,
    u_ref,
    r,
    m;
    nonlinear_snapshots=nothing,
    ε2=hasproperty(u_ref.prob.p, :ε2) ? u_ref.prob.p.ε2 : nothing,
    sigma=hasproperty(u_ref.prob.p, :sigma) ? u_ref.prob.p.sigma : nothing,
    mean_c=hasproperty(u_ref.prob.p, :mean_c) ? u_ref.prob.p.mean_c : mean(u_ref.prob.u0),
    Δx=hasproperty(u_ref.prob.p, :Δx) ? u_ref.prob.p.Δx : nothing,
    Δmeasure=hasproperty(u_ref.prob.p, :Δmeasure) ? u_ref.prob.p.Δmeasure : nothing,
    dimension=1,
    N_obs=10,
    t_obs=collect(LinRange(
        u_ref.prob.tspan[1] + (u_ref.prob.tspan[2] - u_ref.prob.tspan[1]) / (N_obs - 1),
        u_ref.prob.tspan[2],
        N_obs - 1,
    )),
    learner="nn",
    h=8,
    seed=1,
    polynomial_degree=3,
    polynomial_initial_coefficients=nothing,
)
    isnothing(ε2) && error("ε2 must be supplied or present in u_ref.prob.p")
    isnothing(sigma) && error("sigma must be supplied or present in u_ref.prob.p")
    learner = ch_validate_learner(learner)
    dimension = ch_validate_dimension(dimension)
    frames = hcat(u_ref.u...)
    grid_N = ch_grid_size(size(frames, 1), dimension)
    Δmeasure = isnothing(Δmeasure) ?
        (isnothing(Δx) ? one(float(ε2)) : ch_spatial_measure(Δx, dimension)) :
        Δmeasure

    rom = build_ch_pg_deim_rom(D, frames, r, m; ε2, sigma, mean_c, nonlinear_snapshots)

    nn = nothing
    state = nothing
    θ₀ = if learner == "nn"
        rng = MersenneTwister(seed)
        nn = Chain(Dense(1 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
        ps₀, state₀ = Lux.setup(rng, nn)
        state = state₀
        fmap(x -> Float64.(x), ps₀)
    else
        isnothing(polynomial_initial_coefficients) ?
            ch_initial_polynomial_coefficients(polynomial_degree) :
            Float64.(collect(polynomial_initial_coefficients))
    end
    polynomial_degree = learner == "polynomial" ? length(θ₀) - 1 : nothing

    data = (;
        rom,
        nn,
        state,
        learner,
        model_type=learner,
        mean_c,
        u_ref_obs=hcat((u_ref(ti) for ti in t_obs)...),
        t_obs=copy(t_obs),
        Δx,
        Δmeasure,
        dimension,
        boundary_condition="periodic",
        state_shape=dimension == 1 ? (grid_N,) : (grid_N, grid_N),
        N=size(frames, 1),
        grid_N,
        ε2,
        k=nothing,
        sigma,
        full_u₀=copy(u_ref.prob.u0),
        reference_saved_times=copy(u_ref.t),
        reference_algorithm=string(nameof(typeof(u_ref.alg))),
        h=learner == "nn" ? h : nothing,
        activation=learner == "nn" ? "tanh" : "polynomial",
        polynomial_degree,
        polynomial_coefficient_order=learner == "polynomial" ? "ascending powers of u" : nothing,
        polynomial_initial_coefficients=learner == "polynomial" ? copy(θ₀) : nothing,
        seed,
        requested_N_obs=N_obs,
        use_default_nonlinearity=isnothing(nonlinear_snapshots),
    )

    ### ADJUSTED: Store only reduced operators and trainable θ in ROM parameters for AD compatibility.
    p₀ = ComponentVector(Atilde=rom.Atilde, Phi_p=rom.Phi_p, Btilde=rom.Btilde, θ=θ₀)
    a₀ = rom.Phi' * (u_ref.prob.u0 .- mean_c)
    return ch_ROM_problem(a₀, u_ref.prob.tspan, p₀, data)
end


##### CH ROM Optimization #####

"""Materialize CH ROM window references and zero-mean reduced initial states."""
function materialize_CH_ROM_batch(u_ref, rom, mean_c, specs)
    return [
        merge(window, (; u0_rom=rom.Phi' * (window.u0 .- mean_c)))
        for window in materialize_batch(u_ref, specs)
    ]
end

"""Compute one window's reconstructed spatially weighted CH ROM loss."""
function variable_window_CH_ROM_loss(window, prob, p, alg, sensalg, normalization)
    data = prob.f.f.data
    window_prob = remake(prob; u0=window.u0_rom, tspan=(window.t_start, window.t_end), p=p)
    sol = solve(window_prob, alg; saveat=window.t_obs, sensealg=sensalg)
    total = zero(eltype(first(sol.u)))
    @inbounds for j in eachindex(sol.u, window.u_ref_obs)
        u_model = data.mean_c .+ data.rom.Phi * sol.u[j]
        u_ref = window.u_ref_obs[j]
        @simd for i in eachindex(u_model, u_ref)
            total += abs2(u_model[i] - u_ref[i])
        end
    end
    loss = 0.5 * data.Δmeasure * total
    return normalization == "mean" ? loss / length(window.u_ref_obs) : loss
end

"""Average or sum the CH ROM windows scheduled for one optimizer iteration."""
function variable_window_CH_ROM_batch_loss(batch, prob, p, alg, sensalg, normalization)
    total = zero(eltype(first(batch).u0_rom))
    for window in batch
        total += variable_window_CH_ROM_loss(window, prob, p, alg, sensalg, normalization)
    end
    return normalization == "mean" ? total / length(batch) : total
end

"""Run staged Adam training for a CH Petrov-Galerkin DEIM ROM."""
function run_variable_window_CH_ROM_optimization(
    u_ref,
    prob;
    eta=5e-2,
    beta=(0.9, 0.99),
    N_iter=400,
    window_T=nothing,
    window_N_obs=nothing,
    window_start_policy="beginning",
    batch_size=1,
    loss_normalization="mean",
    window_seed=1,
    validation_N_obs=prob.f.f.data.requested_N_obs,
    alg=TRBDF2(autodiff=AutoFiniteDiff()),
    sensalg=GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP()),
    warmup=true,
    save_frequency=nothing,
    print_frequency=10,
)
    data = prob.f.f.data
    core = run_variable_window_stages(
        u_ref,
        prob,
        prob.p;
        optimization_data=(; rom_prob=prob),
        materialize_model_batch=(reference, specs) -> materialize_CH_ROM_batch(reference, data.rom, data.mean_c, specs),
        rebuild_params=(p, re, theta) -> ComponentVector(
            Atilde=p.Atilde,
            Phi_p=p.Phi_p,
            Btilde=p.Btilde,
            θ=re(theta),
        ),
        batch_loss=variable_window_CH_ROM_batch_loss,
        validation_N_obs,
        log_name="run_variable_window_CH_ROM_optimization",
        eta,
        beta,
        N_iter,
        window_T,
        window_N_obs,
        window_start_policy,
        batch_size,
        loss_normalization,
        window_seed,
        alg,
        sensalg,
        warmup,
        save_frequency,
        print_frequency,
    )
    return (;
        core.result,
        core.parameter_history,
        core.final_loss,
        core.final_training_loss,
        core.final_full_trajectory_loss,
        rom_prob=prob,
        run_settings=core.settings,
        core.window_history,
        core.validation_history,
    )
end


##### CH ROM Save Helpers #####

### ADJUSTED: Save CH ROM outputs with the same raw files and metadata field set as AC ROM outputs.
"""
Save the minimal raw data needed to reconstruct a CH ROM optimization under
`Optimization/Data/<run_name>`.
"""
function save_CH_ROM_optimization_data(output, run_name::AbstractString)
    data_root = normpath(joinpath(@__DIR__, "..", "..", "..", "Optimization", "Data"))
    run_directory = assert_run_name_available(run_name; data_root)
    mkpath(data_root)
    mkdir(run_directory)

    prob = output.rom_prob
    data = prob.f.f.data
    rom = data.rom
    learner = hasproperty(data, :learner) ? data.learner : "nn"
    model_type = hasproperty(data, :model_type) ? data.model_type : learner
    polynomial_degree = hasproperty(data, :polynomial_degree) ? data.polynomial_degree : nothing
    polynomial_initial_coefficients = hasproperty(data, :polynomial_initial_coefficients) ? data.polynomial_initial_coefficients : nothing
    polynomial_final_coefficients = model_type == "polynomial" ? copy(last(output.parameter_history).θ) : nothing

    rom_data = (;
        N=data.N,
        grid_N=data.grid_N,
        dimension=data.dimension,
        boundary_condition=data.boundary_condition,
        state_shape=data.state_shape,
        ε2=data.ε2,
        k=data.k,
        sigma=data.sigma,
        mean_c=data.mean_c,
        Δx=data.Δx,
        Δmeasure=data.Δmeasure,
        tspan=prob.tspan,
        t_obs=data.t_obs,
        u₀=data.full_u₀,
        reference_saved_times=data.reference_saved_times,
        r=size(rom.Phi, 2),
        m=size(rom.V, 2),
        spatial_modes=rom.Phi,
        deim_modes=rom.V,
        deim_indices=rom.points,
        state_singular_values=rom.state_singular_values,
        nonlinear_singular_values=rom.nonlinear_singular_values,
        learner,
        model_type,
        polynomial_degree,
        polynomial_coefficient_order=hasproperty(data, :polynomial_coefficient_order) ? data.polynomial_coefficient_order : nothing,
        polynomial_initial_coefficients,
        polynomial_final_coefficients,
        h=data.h,
        activation=hasproperty(data, :activation) ? data.activation : "tanh",
        seed=data.seed,
        use_default_nonlinearity=data.use_default_nonlinearity,
    )

    serialize(joinpath(run_directory, "parameter_history.jls"), output.parameter_history)
    serialize(joinpath(run_directory, "rom_data.jls"), rom_data)

    metadata = (;
        saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        julia_version=VERSION,
        N=rom_data.N,
        grid_N=rom_data.grid_N,
        dimension=rom_data.dimension,
        boundary_condition=rom_data.boundary_condition,
        state_shape=rom_data.state_shape,
        ε2=rom_data.ε2,
        k=rom_data.k,
        sigma=rom_data.sigma,
        mean_c=rom_data.mean_c,
        Δx=rom_data.Δx,
        Δmeasure=rom_data.Δmeasure,
        tspan=rom_data.tspan,
        N_obs=length(rom_data.t_obs),
        reference_steps=length(rom_data.reference_saved_times) - 1,
        r=rom_data.r,
        m=rom_data.m,
        learner=rom_data.learner,
        model_type=rom_data.model_type,
        polynomial_degree=rom_data.polynomial_degree,
        polynomial_final_coefficients=rom_data.polynomial_final_coefficients,
        h=rom_data.h,
        activation=rom_data.activation,
        seed=rom_data.seed,
        reference_algorithm=data.reference_algorithm,
        output.run_settings...,
        initial_loss=first(output.parameter_history).loss,
        final_loss=output.final_loss,
        parameter_snapshots=length(output.parameter_history),
    )

    open(joinpath(run_directory, "metadata.txt"), "w") do io
        for (name, value) in pairs(metadata)
            print(io, name, " = ")
            show(io, value)
            println(io)
        end
    end

    return run_directory
end

### ADJUSTED: Save CH variable-window ROM histories under the same names as AC ROM outputs.
"""Save variable-window CH ROM output under the standard run directory."""
function save_variable_window_CH_ROM_optimization_data(output, run_name::AbstractString)
    run_directory = save_CH_ROM_optimization_data(output, run_name)
    serialize(joinpath(run_directory, "window_history.jls"), output.window_history)
    serialize(joinpath(run_directory, "validation_history.jls"), output.validation_history)
    serialize(joinpath(run_directory, "evaluation_history.jls"), output.validation_history)
    return run_directory
end
