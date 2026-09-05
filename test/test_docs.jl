using REPL   # Base.Docs.doc(::Base.Docs.Binding) is only defined once REPL is loaded

@testset "documentation" begin
    @testset "every exported symbol is documented" begin
        undocumented = String[]
        for name in names(CopyNumberEvolution)
            name === :CopyNumberEvolution && continue
            txt = string(Base.Docs.doc(Base.Docs.Binding(CopyNumberEvolution, name)))
            (isempty(strip(txt)) || occursin("No documentation found", txt)) &&
                push!(undocumented, string(name))
        end
        @test isempty(undocumented)
    end

    @testset "the module itself is documented" begin
        txt = string(Base.Docs.doc(CopyNumberEvolution))
        @test occursin("copy-number", lowercase(txt))
    end

    @testset "the manual mentions every exported symbol at least once" begin
        # api.md uses @autodocs, so this checks the narrative pages, not the reference.
        srcdir = joinpath(@__DIR__, "..", "docs", "src")
        @test isdir(srcdir)
        prose = join([read(joinpath(srcdir, f), String)
                      for f in readdir(srcdir) if endswith(f, ".md")], "\n")
        mentioned(name) = occursin(
            Regex("(?<![A-Za-z0-9_])" * name * "(?![A-Za-z0-9_])"), prose)
        missing_ = [string(n) for n in names(CopyNumberEvolution)
                    if n !== :CopyNumberEvolution && !mentioned(string(n))]
        @test isempty(missing_)
    end
end
