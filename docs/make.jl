using CFDomains
using Documenter

DocMeta.setdocmeta!(CFDomains, :DocTestSetup, :(using CFDomains); recursive=true)

makedocs(;
    modules=[CFDomains],
    authors="The ClimFlows contributors",
    sitename="CFDomains.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/CFDomains.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/CFDomains.jl",
    devbranch="main",
)
