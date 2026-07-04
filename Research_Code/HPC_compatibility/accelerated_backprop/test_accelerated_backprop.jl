### ADJUSTED: Exercise every declared Lux/SimpleChains and VJP pairing end to end locally.
using Test

include(joinpath(@__DIR__, "accelerated_backprop_FOM_hpc.jl"))

@testset "accelerated backprop matrix" begin
    N = 8
    L = 1.0
    ε2 = 1e-2
    k = 1.0
    tfinal = 0.02
    reference = build_ac_reference(;
        N,
        L,
        ε2,
        k,
        tfinal,
        reference_dt_factor=0.5,
    )
    prepared = prepare_for_optimization(;
        N,
        L,
        ε2,
        tspan=reference.tspan,
        N_obs=1,
        u₀=reference.u₀,
        h=8,
        seed=1,
    )

    lux_case = build_backprop_network_case(prepared, "lux")
    simple_case = build_backprop_network_case(prepared, "simplechains")
    window = build_backprop_window(reference.u_ref, 0.0, 0.01, 1)
    benchmark_alg = backprop_benchmark_solver()
    ### ADJUSTED: Keep the original solver mode available for matched Lux benchmarks.
    @test backprop_benchmark_solver("production") isa TRBDF2
    @test_throws ErrorException backprop_benchmark_solver("invalid")
    ### ADJUSTED: Verify the adapted network reproduces the Lux trajectory under the common solver.
    lux_problem = build_backprop_loss(lux_case, window; alg=benchmark_alg, autojacvec=ReverseDiffVJP(true))
    simple_problem = build_backprop_loss(simple_case, window; alg=benchmark_alg, autojacvec=ReverseDiffVJP(true))
    @test lux_problem.loss(lux_problem.theta0) ≈ simple_problem.loss(simple_problem.theta0) rtol=1e-10

    mktempdir() do data_root
        output = run_accelerated_backprop_benchmarks(
            reference,
            prepared;
            networks=["lux", "simplechains"],
            vjps=collect(BACKPROP_VJP_NAMES),
            window_T_values=[0.01],
            window_N_obs_values=[1],
            repeats=1,
            run_name="local_matrix",
            data_root,
        )
        @test length(output.rows) == 2 * length(BACKPROP_VJP_NAMES)
        for row in output.rows
            if row.expected_support
                @test row.status == "ok"
            else
                @test row.status in ("ok", "unsupported")
            end
            if row.status == "ok"
                @test row.relative_gradient_error < 1e-3
                @test row.relative_directional_error < 1e-2
            end
        end
        @test isfile(output.csv_path)
        @test isfile(joinpath(output.run_directory, "benchmark_results.jls"))
        @test isfile(joinpath(output.run_directory, "benchmark_metadata.jls"))
    end
end
