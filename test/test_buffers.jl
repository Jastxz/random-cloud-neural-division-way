# test_buffers.jl — Tests unitarios de gestión de buffers pre-alocados
# Valida: Requisito 10.6

using Test

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
module _StubsBuffers
    struct MapaDeSoluciones{T<:AbstractFloat}
        data::Vector{T}
    end
    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
    include(joinpath(@__DIR__, "..", "src", "buffers.jl"))
end

const crear_buffers = _StubsBuffers.crear_buffers

# ─── Tests ────────────────────────────────────────────────────────────────────

@testset "crear_buffers" begin
    @testset "dimensiones correctas basadas en capa más grande" begin
        bufs = crear_buffers(Float64, [4, 10, 3], 32)
        @test size(bufs.buffer_activacion) == (10, 32)
        @test size(bufs.buffer_gradiente) == (10, 32)
    end

    @testset "tipo numérico Float64" begin
        bufs = crear_buffers(Float64, [5, 8, 2], 16)
        @test eltype(bufs.buffer_activacion) === Float64
        @test eltype(bufs.buffer_gradiente) === Float64
    end

    @testset "tipo numérico Float32 (GPU path)" begin
        bufs = crear_buffers(Float32, [5, 8, 2], 16)
        @test eltype(bufs.buffer_activacion) === Float32
        @test eltype(bufs.buffer_gradiente) === Float32
    end

    @testset "buffers inicializados a cero" begin
        bufs = crear_buffers(Float64, [3, 7, 4], 10)
        @test all(iszero, bufs.buffer_activacion)
        @test all(iszero, bufs.buffer_gradiente)
    end

    @testset "red de una sola capa" begin
        bufs = crear_buffers(Float64, [5], 8)
        @test size(bufs.buffer_activacion) == (5, 8)
        @test size(bufs.buffer_gradiente) == (5, 8)
    end

    @testset "dims_red vacío lanza ArgumentError" begin
        @test_throws ArgumentError crear_buffers(Float64, Int[], 10)
    end

    @testset "n_samples < 1 lanza ArgumentError" begin
        @test_throws ArgumentError crear_buffers(Float64, [4, 8], 0)
        @test_throws ArgumentError crear_buffers(Float64, [4, 8], -1)
    end
end
