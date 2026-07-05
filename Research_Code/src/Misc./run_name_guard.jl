if !isdefined(@__MODULE__, :optimization_data_root)
    """Return the shared optimization Data directory."""
    function optimization_data_root()
        ### ADJUSTED: Resolve saved data from the new src/Misc. location.
        return normpath(joinpath(@__DIR__, "..", "..", "Optimization", "Data"))
    end
end

if !isdefined(@__MODULE__, :assert_run_name_available)
    """Fail if `Optimization/Data/<run_name>` already exists."""
    function assert_run_name_available(run_name::AbstractString; data_root=optimization_data_root())
        run_directory = joinpath(data_root, run_name)
        isdir(run_directory) && error("Run directory already exists: $run_directory")
        return run_directory
    end
end
