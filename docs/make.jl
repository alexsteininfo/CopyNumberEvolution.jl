using Documenter
using CopyNumberEvolution

DocMeta.setdocmeta!(CopyNumberEvolution, :DocTestSetup,
                    :(using CopyNumberEvolution); recursive = true)

makedocs(
    sitename = "CopyNumberEvolution.jl",
    modules = [CopyNumberEvolution],
    authors = "Alexander Stein",
    pages = [
        "Home" => "index.md",
        "Concepts" => "concepts.md",
        "Input: trees" => "trees.md",
        "The alteration model" => "model.md",
        "Output" => "output.md",
        "MEDICC2 interoperability" => "interop.md",
        "Limitations and open questions" => "limitations.md",
        "API reference" => "api.md",
    ],
    doctest = true,
    checkdocs = :exports,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
)
