
if !isdefined(@__MODULE__, :hpc_log)
    function hpc_log(source, message)
        println("[", source, "] ", message)
        flush(stdout)
    end
end


if !isdefined(@__MODULE__, :hpc_log_package)
    function hpc_log_package(package, state)
        hpc_log("package-load", string(state, " ", package))
    end
end


if !isdefined(@__MODULE__, :hpc_log_timed)
    function hpc_log_timed(source, message)
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
        hpc_log(source, "[$timestamp] $message")
    end
end
