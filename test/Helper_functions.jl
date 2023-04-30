using Random
using Test
using LinearAlgebra
using Distributions
using StaticArrays

include(joinpath(@__DIR__, "..", "src", "SDE_inference.jl"))

@testset "Helper functions" begin
    @testset "Systematic resampling" begin
        n_samples = 37088
        index_sampling = Vector{UInt32}(undef, n_samples)
        weights = [0.05, 0.2, 0.05 , 0.1, 0.6]
        systematic_resampling!(index_sampling, weights, n_samples, rand())
        @test sum(index_sampling .== 5) / n_samples ≈ 0.6 atol=1e-3
        @test sum(index_sampling .== 2) / n_samples ≈ 0.2 atol=1e-3
    end

    @testset "Test cholesky" begin
        Random.seed!(123)
        A1 = diagm(rand(5))
        A2 = rand(LKJ(5, 0.2))
        A3 = A2 + Diagonal(A2) .* 1.0
        A4 = rand(LKJ(10, 3.2))

        A1_lower = cholesky(A1).L
        calc_cholesky!(A1, 5)
        @test norm(LowerTriangular(A1) - A1_lower) ≤ 1e-7

        A2_lower = cholesky(A2).L
        calc_cholesky!(A2, 5)
        @test norm(LowerTriangular(A2) - A2_lower) ≤ 1e-7

        A3_lower = cholesky(A3).L
        calc_cholesky!(A3, 5)
        @test norm(LowerTriangular(A3) - A3_lower) ≤ 1e-7

        A4_lower = cholesky(A4).L
        calc_cholesky!(A4, 10)
        @test norm(LowerTriangular(A4) - A4_lower) ≤ 1e-7
    end

    @testset "MVE-normal logpdf" begin
        Random.seed!(123)
        mean_vec = MVector{5, Float64}([1.0, 1.0, 2.0, 0.0, -1.0])
        x_curr = MVector{5, Float64}([0.0, 0.0, 0.0, 0.0, 0.0])
        cov_mat = MMatrix{5, 5, Float64}(rand(LKJ(5, 0.2)))
        dist_ref = MvNormal(mean_vec, cov_mat)
        samples = rand(dist_ref, 10)
        for i in 1:10
            x_curr .= samples[:, i]
            logpdf_val1 = logpdf(dist_ref, x_curr)
            _cov_mat = deepcopy(cov_mat)
            calc_cholesky!(_cov_mat, 5)
            logpdf_val2 = calc_log_pdf_mvn(x_curr, mean_vec, _cov_mat, 5)

            @test logpdf_val1 ≈ logpdf_val2 atol=1e-8
        end
    end
end
    