### ADJUSTED: Extend the existing variable-window FOM path with interchangeable VJP and SimpleChains benchmarks.
include(joinpath(@__DIR__, "..", "Variable_trajectory_length", "variable_window_FOM_opt_AC_hpc.jl"))
include(joinpath(@__DIR__, "..", "hpc_common.jl"))

using Adapt
using Mooncake
using SimpleChains
using Statistics

const BACKPROP_VJP_NAMES = (
    "auto",
    "finite_diff",
    "forward_diff",
    "reverse_diff",
    "reverse_diff_compiled",
    "zygote",
    "tracker",
    "enzyme",
    "enzyme_runtime",
    "mooncake",
    "reactant",
)

"""Return the `GaussAdjoint` VJP setting named by the benchmark CLI."""
function backprop_autojacvec(name::AbstractString)
    normalized = lowercase(strip(name))
    normalized == "auto" && return nothing
    normalized == "finite_diff" && return false
    normalized == "forward_diff" && return true
    normalized == "reverse_diff" && return ReverseDiffVJP(false)
    normalized == "reverse_diff_compiled" && return ReverseDiffVJP(true)
    normalized == "zygote" && return ZygoteVJP()
    normalized == "tracker" && return TrackerVJP()
    normalized == "enzyme" && return EnzymeVJP()
    ### ADJUSTED: Expose Enzyme's runtime-activity workaround and qualify the unexported Mooncake VJP.
    normalized == "enzyme_runtime" && return EnzymeVJP(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
    normalized == "mooncake" && return SciMLSensitivity.MooncakeVJP()
    normalized == "reactant" && return ReactantVJP()
    error("Unknown VJP '$name'; expected one of $(join(BACKPROP_VJP_NAMES, ", "))")
end

"""Return whether the current package versions support a network/VJP/solver-mode combination."""
function expected_backprop_support(
    network::AbstractString,
    vjp::AbstractString,
    solver_autodiff::AbstractString="finite_diff",
)
    normalized_network = lowercase(strip(network))
    normalized_vjp = lowercase(strip(vjp))
    normalized_solver = lowercase(strip(solver_autodiff))
    normalized_vjp in ("forward_diff", "tracker", "enzyme", "reactant") && return false
    normalized_network == "simplechains" && normalized_vjp in ("enzyme_runtime", "mooncake") && return false
    normalized_solver == "production" && normalized_network == "simplechains" && return false
    normalized_solver == "production" && normalized_vjp == "mooncake" && return false
    return true
end

"""Create the same 1-8-8-1 network with SimpleChains kernels behind the Lux interface."""
function simplechains_network_from_lux(nn, lux_theta, seed::Integer)
    adaptor = Lux.ToSimpleChainsAdaptor((SimpleChains.static(1),), true)
    simple_nn = Adapt.adapt(adaptor, nn)
    _, simple_state = Lux.setup(MersenneTwister(seed), simple_nn)
    simple_flat = zeros(Float64, Lux.parameterlength(simple_nn))
    simple_views = SimpleChains.params(simple_nn.layer, simple_flat)

    for (index, layer_name) in enumerate(propertynames(lux_theta))
        lux_layer = getproperty(lux_theta, layer_name)
        copyto!(simple_views[index][1], lux_layer.weight)
        copyto!(simple_views[index][2], lux_layer.bias)
    end

    return (; nn=simple_nn, theta=(params=simple_flat,), state=simple_state)
end

"""Build a Lux or Lux-wrapped SimpleChains neural FOM with identical initial weights."""
function build_backprop_network_case(prepared, network_name::AbstractString)
    normalized = lowercase(strip(network_name))
    if normalized == "lux"
        return (; name=normalized, prob=prepared.prob, p0=prepared.p₀)
    elseif normalized == "simplechains"
        converted = simplechains_network_from_lux(
            prepared.nn,
            prepared.p₀.θ,
            prepared.run_params.seed,
        )
        p0 = ComponentVector(
            ε2=prepared.p₀.ε2,
            Δx=prepared.p₀.Δx,
            θ=converted.theta,
        )
        prob = neural_ODE_prob(
            prepared.u₀,
            prepared.run_params.tspan,
            p0,
            converted.nn,
            converted.state,
        )
        return (; name=normalized, prob, p0)
    end
    error("NETWORKS entries must be lux or simplechains")
end

"""Build one benchmark window using the production window materialization path."""
function build_backprop_window(u_ref, t0, window_T, window_N_obs)
    spec = make_window_spec(
        t0,
        window_T,
        window_N_obs;
        stage=1,
        batch=1,
        window=1,
        policy="beginning",
    )
    return materialize_window(u_ref, spec)
end

"""Create the same mean variable-window loss used by FOM training."""
function build_backprop_loss(network_case, window; alg=TRBDF2(), autojacvec=nothing)
    theta0, re = Optimisers.destructure(network_case.p0.θ)
    sensalg = GaussAdjoint(; autojacvec)
    loss = theta -> begin
        params = ComponentVector(
            ε2=network_case.p0.ε2,
            Δx=network_case.p0.Δx,
            θ=re(theta),
        )
        variable_window_loss(window, network_case.prob, params, alg, sensalg, "mean")
    end
    return (; theta0=copy(theta0), loss)
end

"""Build production TRBDF2 or its finite-difference state-Jacobian benchmark variant."""
function backprop_benchmark_solver(solver_autodiff::AbstractString="finite_diff")
    normalized = lowercase(strip(solver_autodiff))
    normalized == "production" && return TRBDF2()
    ### ADJUSTED: SimpleChains and Mooncake require state Jacobians without ForwardDiff dual arrays.
    normalized == "finite_diff" && return TRBDF2(; autodiff=AutoFiniteDiff())
    error("SOLVER_AUTODIFF must be production or finite_diff")
end

"""Check an adjoint gradient against a central finite difference along its normalized direction."""
function directional_gradient_check(loss, theta, reference_gradient; relative_step=1e-4)
    gradient_norm = norm(reference_gradient)
    gradient_norm > 0 || error("Cannot construct a directional check from a zero reference gradient")
    direction = reference_gradient / gradient_norm
    step = relative_step * max(1.0, norm(theta))
    finite_difference = (loss(theta + step * direction) - loss(theta - step * direction)) / (2 * step)
    return (; direction, finite_difference)
end

"""Time a compiled callable repeatedly and return steady-state statistics."""
function benchmark_callable(f, repeats::Integer)
    samples = Vector{Float64}(undef, repeats)
    for index in eachindex(samples)
        GC.gc()
        samples[index] = @elapsed f()
    end
    allocations = @allocated f()
    return (;
        median=median(samples),
        minimum=minimum(samples),
        maximum=maximum(samples),
        allocations,
    )
end

"""Quote one value for a CSV cell."""
csv_cell(value) = "\"" * replace(string(value), '"' => "\"\"") * "\""

"""Write one benchmark result immediately so completed backends survive later failures."""
function write_backprop_row(io, row)
    ### ADJUSTED: Parenthesize the generator before passing the CSV delimiter.
    println(io, join((csv_cell(value) for value in values(row)), ","))
    flush(io)
end

"""
Benchmark every requested network/VJP pair using the production FOM loss.

The production compiled-ReverseDiff gradient is the pairwise reference, and a
central directional finite difference independently checks its scale. Known
package incompatibilities are recorded as `unsupported`; unexpected failures
are recorded as `error`.
"""
function run_accelerated_backprop_benchmarks(
    reference,
    prepared;
    networks=["lux", "simplechains"],
    vjps=collect(BACKPROP_VJP_NAMES),
    window_T_values=[0.1, 2.0],
    window_N_obs_values=[10, 50],
    repeats=3,
    directional_step=1e-4,
    solver_autodiff="finite_diff",
    run_name="accelerated_backprop",
    data_root=normpath(joinpath(@__DIR__, "..", "..", "Optimization", "Data", "BackpropBenchmarks")),
)
    run_directory = joinpath(data_root, run_name)
    mkpath(run_directory)
    csv_path = joinpath(run_directory, "benchmark_results.csv")
    rows = Any[]
    ### ADJUSTED: Allow a matched comparison against the original production TRBDF2 configuration.
    benchmark_alg = backprop_benchmark_solver(solver_autodiff)

    open(csv_path, "w") do io
        header = (
            network="network",
            vjp="vjp",
            solver_autodiff="solver_autodiff",
            window_T="window_T",
            window_N_obs="window_N_obs",
            expected_support="expected_support",
            status="status",
            loss="loss",
            loss_median_seconds="loss_median_seconds",
            loss_allocations="loss_allocations",
            gradient_norm="gradient_norm",
            relative_gradient_error="relative_gradient_error",
            gradient_directional_derivative="gradient_directional_derivative",
            finite_difference_directional_derivative="finite_difference_directional_derivative",
            relative_directional_error="relative_directional_error",
            first_call_seconds="first_call_seconds",
            gradient_median_seconds="gradient_median_seconds",
            gradient_minimum_seconds="gradient_minimum_seconds",
            gradient_maximum_seconds="gradient_maximum_seconds",
            gradient_allocations="gradient_allocations",
            error="error",
        )
        write_backprop_row(io, header)

        for window_T in window_T_values, window_N_obs in window_N_obs_values
            window = build_backprop_window(
                reference.u_ref,
                reference.tspan[1],
                window_T,
                window_N_obs,
            )

            for network in networks
                network_case = build_backprop_network_case(prepared, network)
                ### ADJUSTED: Keep the production VJP as the reference while using the common benchmark solver.
                reference_started = time()
                reference_problem = build_backprop_loss(
                    network_case,
                    window;
                    alg=benchmark_alg,
                    autojacvec=ReverseDiffVJP(true),
                )
                reference_gradient_call = () -> first(Zygote.gradient(reference_problem.loss, reference_problem.theta0))
                reference_gradient = reference_gradient_call()
                reference_first_call_seconds = time() - reference_started
                loss_value = reference_problem.loss(reference_problem.theta0)
                loss_timing = benchmark_callable(() -> reference_problem.loss(reference_problem.theta0), repeats)
                directional_check = directional_gradient_check(
                    reference_problem.loss,
                    reference_problem.theta0,
                    reference_gradient;
                    relative_step=directional_step,
                )

                for vjp in vjps
                    started = time()
                    support_expected = expected_backprop_support(network, vjp, solver_autodiff)
                    try
                        autojacvec = backprop_autojacvec(vjp)
                        problem = build_backprop_loss(
                            network_case,
                            window;
                            alg=benchmark_alg,
                            autojacvec,
                        )
                        gradient_call = () -> first(Zygote.gradient(problem.loss, problem.theta0))
                        ### ADJUSTED: Reuse the already timed reference call for its own matrix row.
                        is_reference = lowercase(strip(vjp)) == "reverse_diff_compiled"
                        gradient = is_reference ? reference_gradient : gradient_call()
                        first_call_seconds = is_reference ? reference_first_call_seconds : time() - started
                        relative_error = norm(gradient - reference_gradient) / max(norm(reference_gradient), eps())
                        gradient_directional = dot(gradient, directional_check.direction)
                        directional_error = abs(gradient_directional - directional_check.finite_difference) /
                            max(abs(gradient_directional), abs(directional_check.finite_difference), eps())
                        timing = benchmark_callable(gradient_call, repeats)
                        row = (;
                            network,
                            vjp,
                            solver_autodiff,
                            window_T,
                            window_N_obs,
                            expected_support=support_expected,
                            status="ok",
                            loss=loss_value,
                            loss_median_seconds=loss_timing.median,
                            loss_allocations=loss_timing.allocations,
                            gradient_norm=norm(gradient),
                            relative_gradient_error=relative_error,
                            gradient_directional_derivative=gradient_directional,
                            finite_difference_directional_derivative=directional_check.finite_difference,
                            relative_directional_error=directional_error,
                            first_call_seconds,
                            gradient_median_seconds=timing.median,
                            gradient_minimum_seconds=timing.minimum,
                            gradient_maximum_seconds=timing.maximum,
                            gradient_allocations=timing.allocations,
                            error="",
                        )
                        push!(rows, row)
                        write_backprop_row(io, row)
                        println("benchmark ok: network=$network vjp=$vjp T=$window_T Nobs=$window_N_obs median=$(timing.median) relative_error=$relative_error directional_error=$directional_error")
                    catch exception
                        row = (;
                            network,
                            vjp,
                            solver_autodiff,
                            window_T,
                            window_N_obs,
                            expected_support=support_expected,
                            status=support_expected ? "error" : "unsupported",
                            loss=NaN,
                            loss_median_seconds=loss_timing.median,
                            loss_allocations=loss_timing.allocations,
                            gradient_norm=NaN,
                            relative_gradient_error=NaN,
                            gradient_directional_derivative=NaN,
                            finite_difference_directional_derivative=directional_check.finite_difference,
                            relative_directional_error=NaN,
                            first_call_seconds=time() - started,
                            gradient_median_seconds=NaN,
                            gradient_minimum_seconds=NaN,
                            gradient_maximum_seconds=NaN,
                            gradient_allocations=0,
                            error=sprint(showerror, exception),
                        )
                        push!(rows, row)
                        write_backprop_row(io, row)
                        ### ADJUSTED: Distinguish known incompatibilities from unexpected benchmark errors in logs.
                        println("benchmark $(row.status): network=$network vjp=$vjp T=$window_T Nobs=$window_N_obs error=$(row.error)")
                    end
                    flush(stdout)
                end
            end
        end
    end

    metadata = (;
        networks=copy(networks),
        vjps=copy(vjps),
        window_T_values=copy(window_T_values),
        window_N_obs_values=copy(window_N_obs_values),
        repeats,
        directional_step,
        solver_autodiff,
        N=prepared.run_params.N,
        h=prepared.run_params.h,
        network_architecture=prepared.run_params.network_architecture,
        solver=solver_autodiff == "production" ? "TRBDF2()" : "TRBDF2(autodiff=AutoFiniteDiff())",
        sensitivity_algorithm="GaussAdjoint",
        loss_normalization="mean",
        julia_version=VERSION,
    )
    serialize(joinpath(run_directory, "benchmark_results.jls"), rows)
    serialize(joinpath(run_directory, "benchmark_metadata.jls"), metadata)
    return (; rows, metadata, run_directory, csv_path)
end
