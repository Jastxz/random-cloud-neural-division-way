# test_validacion.jl — Tests unitarios de validación de datos de entrada
# Valida: Requisitos 6.2, 6.3

using Test

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
module _StubsValidacion
    struct MapaDeSoluciones{T<:AbstractFloat}
        data::Vector{T}
    end
    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
    include(joinpath(@__DIR__, "..", "src", "validacion.jl"))
end

const validar_datos! = _StubsValidacion.validar_datos!

# ─── Tests ────────────────────────────────────────────────────────────────────

@testset "validar_datos!" begin
    @testset "datos válidos pasan sin error" begin
        datos_x = rand(3, 10)   # 3 features, 10 samples
        datos_y = rand(2, 10)   # 2 outputs, 10 samples
        @test validar_datos!(datos_x, datos_y) === nothing
    end

    @testset "datos válidos con 1 muestra" begin
        datos_x = rand(5, 1)
        datos_y = rand(1, 1)
        @test validar_datos!(datos_x, datos_y) === nothing
    end

    @testset "datos vacíos lanzan ArgumentError" begin
        datos_x = rand(3, 0)   # 0 muestras
        datos_y = rand(2, 0)
        @test_throws ArgumentError validar_datos!(datos_x, datos_y)

        # Verificar mensaje descriptivo
        try
            validar_datos!(datos_x, datos_y)
        catch e
            @test e isa ArgumentError
            @test occursin("vacíos", e.msg)
            @test occursin("0 muestras", e.msg)
        end
    end

    @testset "dimensiones inconsistentes lanzan ArgumentError" begin
        datos_x = rand(3, 10)  # 10 muestras
        datos_y = rand(2, 7)   # 7 muestras — inconsistente
        @test_throws ArgumentError validar_datos!(datos_x, datos_y)

        # Verificar mensaje descriptivo
        try
            validar_datos!(datos_x, datos_y)
        catch e
            @test e isa ArgumentError
            @test occursin("inconsistentes", e.msg) || occursin("Inconsistentes", e.msg)
            @test occursin("10", e.msg)
            @test occursin("7", e.msg)
        end
    end

    @testset "acepta AbstractMatrix (SubArray, view)" begin
        x_full = rand(4, 20)
        y_full = rand(2, 20)
        datos_x = @view x_full[:, 1:10]
        datos_y = @view y_full[:, 1:10]
        @test validar_datos!(datos_x, datos_y) === nothing
    end

    @testset "acepta Float32" begin
        datos_x = rand(Float32, 3, 5)
        datos_y = rand(Float32, 1, 5)
        @test validar_datos!(datos_x, datos_y) === nothing
    end
end
