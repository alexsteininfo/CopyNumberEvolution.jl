const BRANCHLENGTH_MODES = (:divisions, :mutations, :time)

# Mutable staging node used only during parsing.
mutable struct _RawNode
    parent::Union{Int,Nothing}
    children::Vector{Int}
    label::Union{String,Nothing}
    brlen::Union{Float64,Nothing}
end

"""
    parse_newick(str; branchlength) -> PhyloTree

Parse a newick string.

A newick file carries **one** number per edge, but this package needs both real time
and a division count, so you must say which the numbers are. There is deliberately no
default: an implicit convention is the kind of thing that becomes invisible and wrong.

| `branchlength` | field populated | rate rules it enables |
|:---|:---|:---|
| `:divisions` | `edge_divisions` | [`PerDivision`](@ref) |
| `:mutations` | `edge_mutations` | [`FromEdgeMutations`](@ref) |
| `:time` | `birthtime`, by cumulative sum from the root, plus `edge_divisions = 1` | [`PerTime`](@ref), and [`PerDivision`](@ref) treating each edge as one division |

Fields not implied by the chosen mode stay `nothing`, and each rate rule throws a
named error when the field it needs is absent.

Supported syntax: named and unnamed internal nodes, arbitrary arity, single-quoted
labels (with `''` for a literal quote), `[...]` comments, missing branch lengths, and
arbitrary whitespace. Under `:divisions` and `:mutations` a fractional branch length
is rounded and warned about, since a fractional count is usually a sign the file
should have been read as `:time`.

# Examples
```jldoctest
julia> t = parse_newick("((A:1.0,B:2.0)X:0.5,C:3.0)R;"; branchlength = :time);

julia> node(t, node_by_label(t, "A")).birthtime
1.5
```
"""
function parse_newick(str::AbstractString; branchlength::Symbol = :unset)
    branchlength in BRANCHLENGTH_MODES || throw(ArgumentError(
        "branchlength must be one of $BRANCHLENGTH_MODES; got :$branchlength. " *
        "There is no default on purpose — say whether the numbers are divisions, mutations or time."))
    raw = _parse_raw(str)
    return _raw_to_tree(raw, branchlength)
end

"""
    read_newick(path_or_io; branchlength) -> PhyloTree

Read a newick tree from a file path or an `IO`. See [`parse_newick`](@ref) for the
meaning of `branchlength`.
"""
read_newick(io::IO; branchlength::Symbol = :unset) =
    parse_newick(read(io, String); branchlength = branchlength)
read_newick(path::AbstractString; branchlength::Symbol = :unset) =
    open(io -> read_newick(io; branchlength = branchlength), path)

function _parse_raw(str::AbstractString)
    s = collect(str)
    n = length(s)
    nodes = _RawNode[]
    pos = 1

    function skipspace!()
        while pos <= n
            c = s[pos]
            if isspace(c)
                pos += 1
            elseif c == '['
                depth = 0
                while pos <= n
                    s[pos] == '[' && (depth += 1)
                    s[pos] == ']' && (depth -= 1)
                    pos += 1
                    depth == 0 && break
                end
                depth == 0 || throw(ArgumentError("unterminated comment in newick input"))
            else
                return
            end
        end
    end

    function readlabel!()
        skipspace!()
        pos > n && return nothing
        if s[pos] == '\''
            pos += 1
            buf = Char[]
            while true
                pos <= n || throw(ArgumentError("unterminated quoted label in newick input"))
                if s[pos] == '\''
                    if pos + 1 <= n && s[pos + 1] == '\''
                        push!(buf, '\'')
                        pos += 2
                    else
                        pos += 1
                        break
                    end
                else
                    push!(buf, s[pos])
                    pos += 1
                end
            end
            return String(buf)
        end
        start = pos
        while pos <= n && !(s[pos] in ('(', ')', ',', ':', ';', '[')) && !isspace(s[pos])
            pos += 1
        end
        return pos > start ? String(s[start:pos - 1]) : nothing
    end

    function readlength!()
        skipspace!()
        (pos <= n && s[pos] == ':') || return nothing
        pos += 1
        skipspace!()
        start = pos
        while pos <= n && (isdigit(s[pos]) || s[pos] in ('.', '-', '+', 'e', 'E'))
            pos += 1
        end
        pos > start || throw(ArgumentError("expected a number after ':' in newick input"))
        txt = String(s[start:pos - 1])
        val = tryparse(Float64, txt)
        val === nothing && throw(ArgumentError("could not parse branch length '$txt'"))
        val >= 0 || throw(ArgumentError("negative branch length $val is not allowed"))
        return val
    end

    function parsesubtree!(parent::Union{Int,Nothing})
        skipspace!()
        pos <= n || throw(ArgumentError("unexpected end of newick input"))
        push!(nodes, _RawNode(parent, Int[], nothing, nothing))
        me = length(nodes)
        parent === nothing || push!(nodes[parent].children, me)
        if s[pos] == '('
            pos += 1
            skipspace!()
            (pos <= n && s[pos] == ')') &&
                throw(ArgumentError("empty branch set '()' in newick input"))
            while true
                skipspace!()
                (pos <= n && (s[pos] == ',' || s[pos] == ')')) &&
                    throw(ArgumentError(
                        "empty element in a newick branch set at position $pos: a stray, leading or trailing comma"))
                parsesubtree!(me)
                skipspace!()
                pos <= n || throw(ArgumentError("unbalanced parentheses in newick input"))
                if s[pos] == ','
                    pos += 1
                elseif s[pos] == ')'
                    pos += 1
                    break
                else
                    throw(ArgumentError("expected ',' or ')' in newick input, found '$(s[pos])'"))
                end
            end
            isempty(nodes[me].children) && throw(ArgumentError("empty branch set '()' in newick input"))
        end
        nodes[me].label = readlabel!()
        nodes[me].brlen = readlength!()
        return me
    end

    skipspace!()
    pos <= n || throw(ArgumentError("empty newick input"))
    s[pos] == ';' && throw(ArgumentError("empty newick tree ';' has no root node"))
    parsesubtree!(nothing)
    skipspace!()
    (pos <= n && s[pos] == ';') ||
        throw(ArgumentError("newick input must end with ';'" *
            (pos <= n ? " but continues with '$(s[pos])'" : "")))
    pos += 1
    skipspace!()
    pos > n || throw(ArgumentError("trailing content after ';' in newick input"))
    return nodes
end

function _raw_to_tree(raw::Vector{_RawNode}, mode::Symbol)
    n = length(raw)
    order = _raw_preorder(raw)
    birthtimes = Vector{Union{Float64,Nothing}}(nothing, n)
    divisions = Vector{Union{Int,Nothing}}(nothing, n)
    mutations = Vector{Union{Int,Nothing}}(nothing, n)
    fractional = 0

    for i in order
        nd = raw[i]
        if mode === :time
            if nd.parent === nothing
                birthtimes[i] = 0.0
            else
                pbt = birthtimes[nd.parent]
                birthtimes[i] = (pbt === nothing || nd.brlen === nothing) ? nothing : pbt + nd.brlen
                divisions[i] = 1
            end
        elseif nd.parent !== nothing && nd.brlen !== nothing
            v = nd.brlen
            abs(v - round(v)) > 1e-9 && (fractional += 1)
            iv = round(Int, v)
            mode === :divisions ? (divisions[i] = iv) : (mutations[i] = iv)
        end
    end

    fractional > 0 && @warn "newick branch lengths under :$mode should be whole numbers; $fractional fractional value(s) were rounded. If these are real times, read the file with branchlength = :time."

    nodes = [PhyloNode(i, raw[i].parent, raw[i].children,
                       birthtimes[i], divisions[i], mutations[i], raw[i].label, nothing)
             for i in 1:n]
    return PhyloTree(nodes)
end

function _raw_preorder(raw::Vector{_RawNode})
    r = findfirst(nd -> nd.parent === nothing, raw)
    out = Int[]
    stack = [r]
    while !isempty(stack)
        i = pop!(stack)
        push!(out, i)
        append!(stack, raw[i].children)
    end
    return out
end

const _NEWICK_SPECIAL = ('(', ')', ',', ':', ';', '[', ']', '\'', ' ')

_quote_label(name::AbstractString) =
    any(c -> c in _NEWICK_SPECIAL, name) ? "'" * replace(name, "'" => "''") * "'" : name

function _emit_name(t::PhyloTree, i::Integer, labels::Symbol)
    labels === :label && return cellname(t, i)
    labels === :id && return "cell_$(i)"
    labels === :source_id && begin
        sid = node(t, i).source_id
        sid === nothing && throw(ArgumentError("node $i has no source_id, so labels = :source_id cannot name it"))
        return "cell_$(sid)"
    end
    throw(ArgumentError("labels must be :label, :id or :source_id; got :$labels"))
end

function _emit_length(t::PhyloTree, i::Integer, mode::Symbol)
    if mode === :time
        return edge_time(t, i)
    elseif mode === :divisions
        v = node(t, i).edge_divisions
        v === nothing && throw(ArgumentError(
            "cannot write branchlength = :divisions: node $i has no edge_divisions"))
        return v
    else
        v = node(t, i).edge_mutations
        v === nothing && throw(ArgumentError(
            "cannot write branchlength = :mutations: node $i has no edge_mutations"))
        return v
    end
end

"""
    newick_string(tree; branchlength, labels = :label) -> String

Render `tree` as a newick string.

`branchlength` chooses **which** quantity the single branch-length field carries —
`:time`, `:divisions` or `:mutations` — and throws if any non-root node lacks it. The
choice is explicit on write for the same reason it is on read: newick cannot carry
both real time and a division count, and a silent convention is a bug waiting to
happen.

`labels` picks the emitted names: `:label` uses [`cellname`](@ref) (a node's label, or
`cell_<id>`), `:id` always uses `cell_<dense id>`, and `:source_id` uses
`cell_<upstream id>`.
"""
function newick_string(t::PhyloTree; branchlength::Symbol = :unset, labels::Symbol = :label)
    branchlength in BRANCHLENGTH_MODES || throw(ArgumentError(
        "branchlength must be one of $BRANCHLENGTH_MODES; got :$branchlength"))
    labels in (:label, :id, :source_id) || throw(ArgumentError(
        "labels must be :label, :id or :source_id; got :$labels"))
    io = IOBuffer()
    _write_subtree(io, t, treeroot(t), branchlength, labels)
    print(io, ';')
    return String(take!(io))
end

# Iterative emission: a frame is (node, child index, already-opened flag), so a
# 10^5-deep tree writes without recursion.
function _write_subtree(io::IO, t::PhyloTree, start::Int, mode::Symbol, labels::Symbol)
    stack = Tuple{Int,Int}[(start, 0)]
    while !isempty(stack)
        i, k = pop!(stack)
        kids = childrenof(t, i)
        if k == 0 && !isempty(kids)
            print(io, '(')
            push!(stack, (i, 1))
            push!(stack, (kids[1], 0))
        elseif k > 0 && k < length(kids)
            print(io, ',')
            push!(stack, (i, k + 1))
            push!(stack, (kids[k + 1], 0))
        else
            !isempty(kids) && print(io, ')')
            print(io, _quote_label(_emit_name(t, i, labels)))
            if !isroot(t, i)
                print(io, ':', _emit_length(t, i, mode))
            end
        end
    end
    return io
end

"""
    write_newick(io_or_path, tree; branchlength, labels = :label)

Write `tree` in newick format to an `IO` or a file path. See
[`newick_string`](@ref).
"""
write_newick(io::IO, t::PhyloTree; branchlength::Symbol = :unset, labels::Symbol = :label) =
    print(io, newick_string(t; branchlength = branchlength, labels = labels))

write_newick(path::AbstractString, t::PhyloTree; branchlength::Symbol = :unset,
             labels::Symbol = :label) =
    open(io -> write_newick(io, t; branchlength = branchlength, labels = labels), path, "w")
