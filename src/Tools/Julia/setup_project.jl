using Pkg

println("ACTIVE_PROJECT=", Base.active_project())
println("DEPOT_PATH=", DEPOT_PATH)
flush(stdout)

Pkg.instantiate()
Pkg.precompile()
