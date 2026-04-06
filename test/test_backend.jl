# test_backend.jl — Tests unitarios del módulo backend de cómputo
# Valida: Requisitos 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8

using Test

# ─── Stub de dependencias ────────────────────────────────────────────────────
module _StubsBackend
    # Stub mínimo de KernelAbstractions
    module KernelAbstractions
        struct CPU end
        function get_backend(arr)
            return CPU()
        end
        function allocate(backend, T, dims...)
            return zeros(T, dims...)
        end
    end

    # Stub de MapaDeSoluciones (requerido por tipos.jl → Informe_RCND)
    struct MapaDeSoluciones{T<:AbstractFloat}
        data::Vector{T}
    end

    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
    include(joinpath(@__DIR__, "..", "src", "backend.jl"))
end

const seleccionar_backend = _StubsBackend.seleccionar_backend
const transferir_a_dispositivo! = _StubsBackend.transferir_a_dispositivo!
const transferir_a_cpu! = _StubsBackend.transferir_a_cpu!
const gpu_disponible = _StubsBackend.gpu_disponible
const Configuracion_RCND = _StubsBackend.Configuracion_RCND
const ErrorBackend = _StubsBackend.ErrorBackend

# ─── Tests ────────────────────────────────────────────────────────────────────

@testset "Backend de Cómputo" begin

    @testset "seleccionar_backend — :cpu siempre retorna :cpu" begin
        config = Configuracion_RCND(backend_computo=:cpu)
        @test seleccionar_backend(config, 0) === :cpu
        @test seleccionar_backend(config, 100_000) === :cpu
    end

    @testset "seleccionar_backend — :auto con n_params ≤ umbral retorna :cpu" begin
        config = Configuracion_RCND(backend_computo=:auto, umbral_gpu=10_000)
        @test seleccionar_backend(config, 5_000) === :cpu
        @test seleccionar_backend(config, 10_000) === :cpu  # Exactamente en el umbral → :cpu
        @test seleccionar_backend(config, 0) === :cpu
        @test seleccionar_backend(config, 1) === :cpu
    end

    @testset "seleccionar_backend — :auto con n_params > umbral retorna :gpu" begin
        config = Configuracion_RCND(backend_computo=:auto, umbral_gpu=10_000)
        @test seleccionar_backend(config, 10_001) === :gpu
        @test seleccionar_backend(config, 100_000) === :gpu
    end

    @testset "seleccionar_backend — :auto con umbral_gpu=0, cualquier n_params > 0 retorna :gpu" begin
        config = Configuracion_RCND(backend_computo=:auto, umbral_gpu=0)
        @test seleccionar_backend(config, 1) === :gpu
        @test seleccionar_backend(config, 0) === :cpu  # 0 no es > 0
    end

    @testset "seleccionar_backend — :gpu sin GPU lanza ErrorBackend" begin
        # En entorno de test sin GPU, gpu_disponible() retorna false
        config = Configuracion_RCND(backend_computo=:gpu)
        @test_throws ErrorBackend seleccionar_backend(config, 100)

        try
            seleccionar_backend(config, 100)
        catch e
            @test e isa ErrorBackend
            @test e.backend === :gpu
            @test occursin("GPU", e.mensaje) || occursin("gpu", e.mensaje)
        end
    end

    @testset "seleccionar_backend — backend desconocido lanza ErrorBackend" begin
        config = Configuracion_RCND(backend_computo=:tpu)
        @test_throws ErrorBackend seleccionar_backend(config, 100)
    end

    @testset "gpu_disponible — retorna false sin GPU real" begin
        # En entorno de test estándar sin GPU
        @test gpu_disponible() == false
    end

    @testset "transferir_a_dispositivo! — :cpu mantiene tipo si ya es correcto" begin
        datos = randn(Float64, 3, 5)
        resultado = transferir_a_dispositivo!(datos, :cpu, Float64)
        @test resultado === datos  # Mismo objeto, sin copia
        @test eltype(resultado) == Float64
    end

    @testset "transferir_a_dispositivo! — :cpu convierte tipo si es diferente" begin
        datos = randn(Float64, 3, 5)
        resultado = transferir_a_dispositivo!(datos, :cpu, Float32)
        @test eltype(resultado) == Float32
        @test size(resultado) == (3, 5)
        @test resultado ≈ Float32.(datos)
    end

    @testset "transferir_a_dispositivo! — :cpu con vectores de arrays" begin
        datos = [randn(Float64, 4, 3), randn(Float64, 2, 4)]
        resultado = transferir_a_dispositivo!(datos, :cpu, Float32)
        @test length(resultado) == 2
        @test eltype(resultado[1]) == Float32
        @test eltype(resultado[2]) == Float32
        @test size(resultado[1]) == (4, 3)
        @test size(resultado[2]) == (2, 4)
    end

    @testset "transferir_a_dispositivo! — :cpu con NamedTuple" begin
        datos = (buffer_activacion=randn(Float64, 5, 3), buffer_gradiente=randn(Float64, 5, 3))
        resultado = transferir_a_dispositivo!(datos, :cpu, Float32)
        @test eltype(resultado.buffer_activacion) == Float32
        @test eltype(resultado.buffer_gradiente) == Float32
    end

    @testset "transferir_a_cpu! — array CPU se retorna como Array" begin
        datos = randn(Float64, 3, 5)
        resultado = transferir_a_cpu!(datos)
        @test resultado isa Array
        @test resultado ≈ datos
    end

    @testset "transferir_a_cpu! — vector de arrays" begin
        datos = [randn(Float64, 4, 3), randn(Float64, 2, 4)]
        resultado = transferir_a_cpu!(datos)
        @test length(resultado) == 2
        @test all(r -> r isa Array, resultado)
        @test resultado[1] ≈ datos[1]
        @test resultado[2] ≈ datos[2]
    end

    @testset "transferir_a_cpu! — NamedTuple" begin
        datos = (buf_a=randn(Float64, 5, 3), buf_b=randn(Float64, 5, 3))
        resultado = transferir_a_cpu!(datos)
        @test resultado.buf_a isa Array
        @test resultado.buf_b isa Array
        @test resultado.buf_a ≈ datos.buf_a
    end

    @testset "seleccionar_backend — valores por defecto de config (Req 10.9)" begin
        config = Configuracion_RCND()  # Defaults: :auto, umbral_gpu=10_000
        @test config.backend_computo === :auto
        @test config.umbral_gpu == 10_000
        # Con pocos parámetros → :cpu
        @test seleccionar_backend(config, 5_000) === :cpu
        # Con muchos parámetros → :gpu
        @test seleccionar_backend(config, 15_000) === :gpu
    end
end
