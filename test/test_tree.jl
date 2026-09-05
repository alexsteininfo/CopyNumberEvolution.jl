@testset "tree" begin
    # Fixture shape (ids in parentheses):
    #            1
    #          /   \
    #         2     3
    #        / \   / \
    #       4   5 6   7
    balanced() = phylotree([nothing, 1, 1, 2, 2, 3, 3];
                           birthtimes = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                           edge_divisions = [nothing, 1, 1, 1, 1, 1, 1],
                           edge_mutations = [nothing, 4, 7, 2, 0, 5, 3],
                           labels = [nothing, nothing, nothing, "a", "b", "c", "d"],
                           source_ids = [10, 20, 30, 40, 50, 60, 70])

    @testset "construction and basic accessors" begin
        t = balanced()
        @test nnodes(t) == 7
        @test treeroot(t) == 1
        @test leaves(t) == [4, 5, 6, 7]
        @test internal_nodes(t) == [1, 2, 3]
        @test childrenof(t, 1) == [2, 3]
        @test parentof(t, 4) == 2
        @test parentof(t, 1) === nothing
        @test isroot(t, 1) && !isroot(t, 2)
        @test isleaf(t, 4) && !isleaf(t, 2)
        @test depth(t, 1) == 0 && depth(t, 4) == 2
        @test node(t, 4).id == 4
        @test node(t, 4).label == "a"
        @test node(t, 1).parent === nothing
    end

    @testset "traversal orders" begin
        t = balanced()
        @test preorder(t) == [1, 2, 4, 5, 3, 6, 7]
        @test postorder(t) == [4, 5, 2, 6, 7, 3, 1]
        @test sort(preorder(t)) == collect(1:7)
    end

    @testset "ancestors and descendant leaves" begin
        t = balanced()
        @test ancestors(t, 4) == [1, 2, 4]
        @test ancestors(t, 1) == [1]
        @test descendant_leaves(t, 2) == [4, 5]
        @test descendant_leaves(t, 1) == [4, 5, 6, 7]
        @test descendant_leaves(t, 4) == [4]
    end

    @testset "mrca" begin
        t = balanced()
        @test mrca(t, [4, 5]) == 2
        @test mrca(t, [4, 6]) == 1
        @test mrca(t, [4]) == 4
        @test mrca(t, [4, 4]) == 4
        @test mrca(t, [4, 5, 6, 7]) == 1
        @test mrca(t, [2, 4]) == 2
        @test_throws ArgumentError mrca(t, Int[])
    end

    @testset "edge_time" begin
        t = balanced()
        @test edge_time(t, 2) ≈ 1.0
        @test edge_time(t, 4) ≈ 1.0
        @test edge_time(t, 5) ≈ 1.5
        @test_throws ArgumentError edge_time(t, 1)          # the root has no incoming edge
        u = phylotree([nothing, 1])                          # no birthtimes at all
        @test_throws ArgumentError edge_time(u, 2)
    end

    @testset "identity helpers" begin
        t = balanced()
        @test node_by_source_id(t, 60) == 6
        @test node_by_label(t, "b") == 5
        @test_throws ArgumentError node_by_source_id(t, 999)
        @test_throws ArgumentError node_by_label(t, "zzz")
        @test cellname(t, 4) == "a"
        @test cellname(t, 2) == "cell_2"
        @test cellname(t, 2; prefix = "node") == "node_2"
    end

    @testset "malformed trees are rejected" begin
        @test_throws ArgumentError phylotree([nothing, nothing])       # two roots
        @test_throws ArgumentError phylotree([1, 1])                   # no root
        @test_throws ArgumentError phylotree([nothing, 5])             # parent out of range
        @test_throws ArgumentError phylotree([nothing, 2, 3, 2])       # cycle: 2->3->2 unreachable
        @test_throws ArgumentError phylotree([nothing, 1]; birthtimes = [0.0])  # length mismatch
    end

    @testset "the constructor's reachability and cycle checks fire" begin
        # Unreachable component: nodes 3 and 4 point at each other, neither
        # descends from the root. No self-parents, so phylotree's own guard
        # does not mask the constructor's reachability walk.
        @test_throws ArgumentError phylotree([nothing, 1, 4, 3])

        # A node listed twice as a child cannot be built through phylotree —
        # it buckets each node under exactly one parent — so construct the
        # nodes directly. This is the path Task 6's newick parser uses.
        dup = [CopyNumberEvolution.PhyloNode(1, nothing, [2, 2], nothing, nothing,
                                             nothing, nothing, nothing),
               CopyNumberEvolution.PhyloNode(2, 1, Int[], nothing, nothing,
                                             nothing, nothing, nothing)]
        @test_throws ArgumentError CopyNumberEvolution.PhyloTree(dup)

        # A child whose recorded parent does not list it. Node 2 claims parent 1,
        # but node 1's child list is empty, so the mutual-consistency check fires.
        # (Giving node 2 a `nothing` parent instead would read as a second root and
        # short-circuit at the root-count check before reaching this branch.)
        bad = [CopyNumberEvolution.PhyloNode(1, nothing, Int[], nothing, nothing,
                                             nothing, nothing, nothing),
               CopyNumberEvolution.PhyloNode(2, 1, Int[], nothing, nothing,
                                             nothing, nothing, nothing)]
        @test_throws ArgumentError CopyNumberEvolution.PhyloTree(bad)
    end

    @testset "non-binary and unary trees are supported" begin
        # a multifurcation and a unary chain, both of which newick and sampled
        # lineage trees produce
        t = phylotree([nothing, 1, 1, 1])
        @test childrenof(t, 1) == [2, 3, 4]
        @test leaves(t) == [2, 3, 4]
        @test preorder(t) == [1, 2, 3, 4]
        @test postorder(t) == [2, 3, 4, 1]
        u = phylotree([nothing, 1, 2, 3])
        @test leaves(u) == [4]
        @test depth(u, 4) == 3
        @test preorder(u) == [1, 2, 3, 4]
    end

    @testset "deep trees do not overflow the stack" begin
        n = 50_000
        parents = Vector{Union{Int,Nothing}}(undef, n)
        parents[1] = nothing
        for i in 2:n
            parents[i] = i - 1
        end
        t = phylotree(parents)
        @test length(preorder(t)) == n
        @test depth(t, n) == n - 1
        @test descendant_leaves(t, 1) == [n]
    end
end
