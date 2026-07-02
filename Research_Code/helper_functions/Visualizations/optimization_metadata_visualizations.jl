### ADJUSTED: Add lightweight metadata-only helpers for saved optimization runs.

"""Print the raw `metadata.txt` for a saved optimization run."""
function print_metadata(run_dir::AbstractString)
    metadata_path = joinpath(run_dir, "metadata.txt")
    println(read(metadata_path, String))
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
