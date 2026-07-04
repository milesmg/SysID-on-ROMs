### ADJUSTED: Add a focused ROM function test for the selected finite-difference/Enzyme setup.
using Test

const REPO_ROOT_ROM_ENZYME_TEST = normpath(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(REPO_ROOT_ROM_ENZYME_TEST, "Research_Code", "helper_functions", "HPC", "ROM_opt_AC_hpc.jl"))
include(joinpath(REPO_ROOT_ROM_ENZYME_TEST, "Research_Code", "HPC_compatibility", "hpc_common.jl"))

"""Return the accelerated ODE and sensitivity algorithms selected by the FOM benchmark."""
function accelerated_rom_algorithms()
    alg = TRBDF2(; autodiff=AutoFiniteDiff())
    sensalg = GaussAdjoint(
        autojacvec=EnzymeVJP(
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        ),
    )
    return (; alg, sensalg)
end

"""Build a small production-structured POD/DEIM ROM for the function test."""
function build_rom_enzyme_test_problem(;
    N=16,
    L=1.0,
    ε2=1e-2,
    k=1.0,
    tfinal=0.05,
    reference_dt_factor=0.05,
    N_obs=3,
    r=3,
    m=3,
    h=8,
    seed=1,
)
    reference = build_ac_reference(; N, L, ε2, k, tfinal, reference_dt_factor)
    A = get_lap1d_matrix(N, ε2, reference.Δx)
    rom_prob = prepare_ROM_optimization(A, reference.u_ref, r, m; N_obs, h, seed)
    algorithms = accelerated_rom_algorithms()
    optprob = set_up_ROM_optimization(
        rom_prob;
        alg=algorithms.alg,
        sensalg=algorithms.sensalg,
    )
    return (; reference, rom_prob, optprob, algorithms)
end

"""Run loss, gradient, directional-derivative, and optimizer-update checks."""
function run_rom_enzyme_function_test(; directional_step=1e-4, kwargs...)
    setup = build_rom_enzyme_test_problem(; kwargs...)
    loss = θ -> setup.optprob.f(θ, setup.optprob.p)
    θ0 = setup.optprob.u0

    initial_loss = loss(θ0)
    gradient = first(Zygote.gradient(loss, θ0))
    direction = gradient / norm(gradient)
    step = directional_step * max(1.0, norm(θ0))
    finite_difference = (loss(θ0 + step * direction) - loss(θ0 - step * direction)) / (2step)
    gradient_directional = dot(gradient, direction)
    directional_error = abs(gradient_directional - finite_difference) /
        max(abs(gradient_directional), abs(finite_difference), eps(Float64))

    optimization = run_ROM_optimization(
        setup.optprob;
        η=1e-3,
        ### ADJUSTED: Optimization.jl uses the first iteration for initial evaluation, so two iterations test one update.
        N_iter=2,
        warmup=false,
        save_frequency=1,
        print_frequency=1,
    )

    return (;
        initial_loss,
        gradient,
        gradient_norm=norm(gradient),
        finite_difference,
        gradient_directional,
        directional_error,
        final_loss=optimization.final_loss,
        parameter_change=norm(optimization.result.u - θ0),
        optimization,
        setup,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "ROM finite-difference solver with runtime Enzyme VJP" begin
        result = run_rom_enzyme_function_test()
        @test isfinite(result.initial_loss)
        @test all(isfinite, result.gradient)
        @test result.gradient_norm > 0
        @test result.directional_error < 1e-2
        @test isfinite(result.final_loss)
        @test result.parameter_change > 0
    end
end
