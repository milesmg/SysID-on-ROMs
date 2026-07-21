# Work with (hyperreduced) ROM data, and project things into and out of ROM

"""Return the fixed 500-time reference snapshot grid used to build every POD/DEIM basis."""
reference_save_times(tfinal) = collect(LinRange(0.0, tfinal, 500))

"""Get first r/m POD/DEIM modes, where r,m passed in as rank. Also return the singular values"""
function pod_modes(frames, rank::Integer)
    result = svd(frames; full=false)
    result.U[:, 1:rank], result.S
end

"""Squared singular value capture. """
pod_capture(singular_values, rank::Integer) = sum(abs2, singular_values[1:rank]) / sum(abs2, singular_values)

"""Get DEIM points
Args:
    - modes = the DEIM snapshots. Size of this matrix gives 'm'
NOTE: At some point, I'll probably want to be able to split the DEIM points somewhat evenly between the functions; right now, it treats each [s1 s2] snapshot as a single vector and does the greedy alg. on the combined.
NOTE on the note: this function can stay the same, regardless. It's just what I pass in
"""
function deim_indices(modes)
    points = Vector{Int}(undef, size(modes, 2))
    points[1] = argmax(abs.(modes[:, 1]))
    for k in 2:length(points)
        basis, selected = modes[:, 1:k-1], points[1:k-1]
        points[k] = argmax(abs.(modes[:, k] - basis * (basis[selected, :] \ modes[selected, k])))
    end
    points
end

"""
This builds a matrix that interpolates function values into the reduced basis based on sparse evaluation at m points. 
Args:
- modes: the DEIM modes; the left singular vectors of the snapshot matrix
- points: the DEIM points; where are we evaluating?
- basis: the state POD basis
"""
deim_projection(modes, points, basis) = (modes[points, :]' \ (modes' * basis))'
"""Project full state down into POD basis
Args:
- basis: the POD basis
- state: the full state
"""
rom_project(basis, state) = basis' * state
"""Project up from the ROM reduced basis into the full state 
Args:
- basis: the POD basis
- state: the full state
"""
rom_reconstruct(basis, state) = basis * state
