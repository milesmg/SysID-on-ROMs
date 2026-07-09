### ADJUSTED: Add lightweight metadata-only helpers for saved optimization runs.
using Serialization

### ADJUSTED: Infer serialized dimension metadata for older runs whose metadata.txt omitted it.
"""Return compact metadata inferred from serialized ROM/FOM data."""
function serialized_metadata_summary(run_dir::AbstractString)
    data_path = isfile(joinpath(run_dir, "rom_data.jls")) ?
        joinpath(run_dir, "rom_data.jls") :
        joinpath(run_dir, "run_params.jls")
    isfile(data_path) || return (;)
    data = deserialize(data_path)
    state_length = if hasproperty(data, :spatial_modes)
        size(data.spatial_modes, 1)
    elseif hasproperty(data, :u₀)
        length(data.u₀)
    else
        hasproperty(data, :N) ? Int(data.N) : 0
    end
    dimension = if hasproperty(data, :dimension)
        Int(data.dimension)
    elseif occursin("2d", lowercase(run_dir))
        2
    elseif hasproperty(data, :N) && Int(data.N)^2 == state_length
        2
    else
        1
    end
    grid_N = if hasproperty(data, :grid_N)
        Int(data.grid_N)
    elseif hasproperty(data, :state_shape)
        Int(first(data.state_shape))
    elseif dimension == 2 && hasproperty(data, :N) && Int(data.N)^2 == state_length
        Int(data.N)
    elseif dimension == 2
        round(Int, sqrt(state_length))
    else
        hasproperty(data, :N) ? Int(data.N) : state_length
    end
    boundary_condition = hasproperty(data, :boundary_condition) ?
        string(data.boundary_condition) :
        (occursin("periodic", lowercase(run_dir)) ? "periodic" : "homogeneous_dirichlet")
    ### ADJUSTED: Include learned nonlinearity metadata for polynomial runs.
    model_type = hasproperty(data, :model_type) ? string(data.model_type) :
        (hasproperty(data, :learner) ? string(data.learner) : "nn")
    polynomial_degree = hasproperty(data, :polynomial_degree) ? data.polynomial_degree : nothing
    return (;
        serialized_file=basename(data_path),
        dimension,
        boundary_condition,
        grid_N,
        state_length,
        model_type,
        polynomial_degree,
    )
end

### ADJUSTED: Print compact derived metadata after raw metadata.txt for 2D run clarity.
"""Print serialized metadata fields that may be missing from older `metadata.txt` files."""
function print_serialized_metadata_summary(run_dir::AbstractString)
    summary = serialized_metadata_summary(run_dir)
    isempty(pairs(summary)) && return nothing
    println("Derived serialized metadata:")
    for (name, value) in pairs(summary)
        print(name, " = ")
        show(value)
        println()
    end
    return nothing
end

"""Print the raw `metadata.txt` for a saved optimization run."""
function print_metadata(run_dir::AbstractString)
    metadata_path = joinpath(run_dir, "metadata.txt")
    println(read(metadata_path, String))
    print_serialized_metadata_summary(run_dir)
    return metadata_path
end


"""Return raw metadata values from `metadata.txt` as strings."""
function read_metadata_values(run_dir::AbstractString)
    metadata = Dict{String, String}()
    metadata_path = joinpath(run_dir, "metadata.txt")
    isfile(metadata_path) || return metadata
    for line in eachline(metadata_path)
        parts = split(line, " = "; limit=2)
        length(parts) == 2 || continue
        metadata[parts[1]] = parts[2]
    end
    return metadata
end


"""Print metadata for a saved ROM optimization run."""
function visualize_ROM_metadata(run_dir::AbstractString)
    ### ADJUSTED: Keep metadata display independent of heavy plotting and solver packages.
    print_metadata(run_dir)
    return nothing
end


"""Print metadata for a saved FOM optimization run."""
function visualize_FOM_metadata(run_dir::AbstractString)
    ### ADJUSTED: Keep metadata display independent of heavy plotting and solver packages.
    print_metadata(run_dir)
    return nothing
end
