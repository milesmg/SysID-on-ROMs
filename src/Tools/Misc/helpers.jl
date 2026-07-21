#### Misc. General Tools ###

function help(f)
    doc = Base.Docs.doc(f)
    show(stdout, MIME"text/plain"(), doc)
    println()
    return nothing
end
