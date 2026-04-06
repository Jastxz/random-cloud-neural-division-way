# test_adaptador.jl — Tests unitarios del módulo adaptador
# Valida: Requisitos 3.1, 3.2, 3.3, 3.4

using Test

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
module _StubsAdaptador
    # Stub de MapaDeSoluciones (DivisionNeuronal.jl)
    struct MapaDeSoluciones{T<:AbstractFloat}
        data::Vector{T}
    end

    # Stub de RedNeuronal (RandomCloud.jl)
    struct RedNeuronal
        capas::Vector{Int}
        pesos::Vector{<:AbstractMatrix}
        biases::Vector{<:AbstractVector}
        activaciones::Vector{Symbol}
    end

    # Stub de RedBase{T} (DivisionNeuronal.jl)
    struct RedBase{T<:AbstractFloat}
        capas::Vector{Int}
        pesos::Vector{Matrix{T}}
        biases::Vector{Vector{T}}
        activaciones::Vector{Symbol}
    end

    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
    include(joinpath(@__DIR__, "..", "src", "adaptador.jl"))
end

const adaptar_red = _StubsAdaptador.adaptar_red
const contar_parametros = _StubsAdaptador.contar_parametros
const validar_arquitectura = _StubsAdaptador.validar_arquitectura
const RedNeuronal = _StubsAdaptador.RedNeuronal
const RedBase = _StubsAdaptador.RedBase
const ErrorAdaptador = _StubsAdaptador.ErrorAdaptador

# ─── Helpers ──────────────────────────────────────────────────────────────────

"""Crea una RedNeuronal válida con dimensiones dadas."""
function crear_red_valida(capas::Vector{Int}; T=Float64)
    n = length(capas) - 1
    pesos = [randn(T, capas[i+1], capas[i]) for i in 1:n]
    biases = [randn(T, capas[i+1]) for i in 1:n]
    activaciones = [[:sigmoid, :relu, :identity][mod1(i, 3)] for i in 1:n]
    RedNeuronal(capas, pesos, biases, activaciones)
end

# ─── Tests ────────────────────────────────────────────────────────────────────

@testset "Adaptador" begin

    @testset "adaptar_red — conversión básica Float64" begin
        red = crear_red_valida([4, 10, 3])
        resultado = adaptar_red(red, Float64)

        @test resultado isa RedBase{Float64}
        @test resultado.capas == [4, 10, 3]
        @test length(resultado.pesos) == 2
        @test length(resultado.biases) == 2
        @test resultado.activaciones == red.activaciones
        @test resultado.pesos[1] ≈ red.pesos[1]
        @test resultado.pesos[2] ≈ red.pesos[2]
        @test resultado.biases[1] ≈ red.biases[1]
        @test resultado.biases[2] ≈ red.biases[2]
    end

    @testset "adaptar_red — conversión Float64 → Float32" begin
        red = crear_red_valida([3, 5, 2]; T=Float64)
        resultado = adaptar_red(red, Float32)

        @test resultado isa RedBase{Float32}
        @test eltype(resultado.pesos[1]) == Float32
        @test eltype(resultado.pesos[2]) == Float32
        @test eltype(resultado.biases[1]) == Float32
        @test eltype(resultado.biases[2]) == Float32
        # Valores preservados (hasta precisión Float32)
        @test resultado.pesos[1] ≈ Float32.(red.pesos[1])
        @test resultado.biases[1] ≈ Float32.(red.biases[1])
    end

    @testset "adaptar_red — red con una sola capa oculta" begin
        red = crear_red_valida([2, 1])
        resultado = adaptar_red(red, Float64)

        @test resultado.capas == [2, 1]
        @test length(resultado.pesos) == 1
        @test size(resultado.pesos[1]) == (1, 2)
    end

    @testset "adaptar_red — red profunda (4 capas ocultas)" begin
        red = crear_red_valida([5, 10, 8, 6, 4, 2])
        resultado = adaptar_red(red, Float64)

        @test resultado.capas == [5, 10, 8, 6, 4, 2]
        @test length(resultado.pesos) == 5
        @test length(resultado.biases) == 5
        @test length(resultado.activaciones) == 5
    end

    @testset "adaptar_red — preserva funciones de activación" begin
        capas = [3, 4, 2]
        pesos = [randn(4, 3), randn(2, 4)]
        biases = [randn(4), randn(2)]
        activaciones = [:relu, :sigmoid]
        red = RedNeuronal(capas, pesos, biases, activaciones)

        resultado = adaptar_red(red, Float64)
        @test resultado.activaciones == [:relu, :sigmoid]
    end

    @testset "adaptar_red — no muta la red original" begin
        red = crear_red_valida([3, 5, 2])
        pesos_orig = deepcopy(red.pesos)
        biases_orig = deepcopy(red.biases)

        adaptar_red(red, Float32)

        @test red.pesos[1] == pesos_orig[1]
        @test red.biases[1] == biases_orig[1]
    end

    @testset "validar_arquitectura — red con menos de 2 capas" begin
        red = RedNeuronal([5], Matrix{Float64}[], Vector{Float64}[], Symbol[])
        @test_throws ErrorAdaptador validar_arquitectura(red)
    end

    @testset "validar_arquitectura — pesos con dimensiones incorrectas" begin
        capas = [3, 4, 2]
        pesos = [randn(4, 3), randn(3, 4)]  # Segunda debería ser (2, 4)
        biases = [randn(4), randn(2)]
        activaciones = [:relu, :sigmoid]
        red = RedNeuronal(capas, pesos, biases, activaciones)

        @test_throws ErrorAdaptador adaptar_red(red, Float64)

        try
            adaptar_red(red, Float64)
        catch e
            @test e isa ErrorAdaptador
            @test e.capa == 2
            @test occursin("pesos", e.mensaje) || occursin("Dimensiones", e.mensaje)
        end
    end

    @testset "validar_arquitectura — biases con dimensión incorrecta" begin
        capas = [3, 4, 2]
        pesos = [randn(4, 3), randn(2, 4)]
        biases = [randn(4), randn(5)]  # Segundo debería ser 2
        activaciones = [:relu, :sigmoid]
        red = RedNeuronal(capas, pesos, biases, activaciones)

        @test_throws ErrorAdaptador adaptar_red(red, Float64)

        try
            adaptar_red(red, Float64)
        catch e
            @test e isa ErrorAdaptador
            @test e.capa == 2
            @test occursin("biases", e.mensaje) || occursin("bias", e.mensaje)
        end
    end

    @testset "validar_arquitectura — número incorrecto de matrices de pesos" begin
        capas = [3, 4, 2]
        pesos = [randn(4, 3)]  # Falta una matriz
        biases = [randn(4), randn(2)]
        activaciones = [:relu, :sigmoid]
        red = RedNeuronal(capas, pesos, biases, activaciones)

        @test_throws ErrorAdaptador adaptar_red(red, Float64)
    end

    @testset "validar_arquitectura — número incorrecto de activaciones" begin
        capas = [3, 4, 2]
        pesos = [randn(4, 3), randn(2, 4)]
        biases = [randn(4), randn(2)]
        activaciones = [:relu]  # Falta una
        red = RedNeuronal(capas, pesos, biases, activaciones)

        @test_throws ErrorAdaptador adaptar_red(red, Float64)
    end

    @testset "contar_parametros — red simple" begin
        # Red [3, 4, 2]: pesos = 3*4 + 4*2 = 20, biases = 4 + 2 = 6, total = 26
        red = crear_red_valida([3, 4, 2])
        @test contar_parametros(red) == 26
    end

    @testset "contar_parametros — red profunda" begin
        # Red [5, 10, 8, 3]: pesos = 5*10 + 10*8 + 8*3 = 50+80+24 = 154
        # biases = 10 + 8 + 3 = 21, total = 175
        red = crear_red_valida([5, 10, 8, 3])
        @test contar_parametros(red) == 175
    end

    @testset "contar_parametros — red mínima" begin
        # Red [2, 1]: pesos = 2*1 = 2, biases = 1, total = 3
        red = crear_red_valida([2, 1])
        @test contar_parametros(red) == 3
    end

    @testset "contar_parametros — funciona con RedBase también (duck-typing)" begin
        red_orig = crear_red_valida([4, 6, 3])
        red_base = adaptar_red(red_orig, Float64)
        @test contar_parametros(red_base) == contar_parametros(red_orig)
    end
end
