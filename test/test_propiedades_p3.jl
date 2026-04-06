# Feature: neural-cloud-division, Property 3: El adaptador preserva todos los atributos de la red
# **Valida: Requisitos 3.1, 3.2, 3.3**

using Test
using Supposition
using Supposition.Data

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
module _StubsP3
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

const adaptar_red_p3 = _StubsP3.adaptar_red
const RedNeuronal_p3 = _StubsP3.RedNeuronal
const RedBase_p3 = _StubsP3.RedBase

# ─── Generadores ──────────────────────────────────────────────────────────────

# Activaciones válidas
const ACTIVACIONES_VALIDAS = [:sigmoid, :relu, :identity]

# Generador de una activación aleatoria
const gen_activacion = Data.SampledFrom(ACTIVACIONES_VALIDAS)

# Generador de dimensión de capa (1-20 neuronas)
const gen_dim_capa = Data.Integers(1, 20)

# Generador de número de capas (2-5 capas totales, es decir 1-4 conexiones)
const gen_n_capas = Data.Integers(2, 5)

# Generador de Float64 finitos para pesos/biases
const gen_peso = Data.Satisfying(Data.Floats{Float64}(), x -> isfinite(x) && abs(x) < 1e6)

"""
Genera una RedNeuronal válida con dimensiones, pesos, biases y activaciones aleatorios.
Usa @composed con un argumento dummy (n_capas_total) para satisfacer la API de Supposition.jl.
"""
const gen_red_neuronal = @composed function gen_red(n_capas_total = gen_n_capas)
    # Generar dimensiones de cada capa
    dims = [produce!(gen_dim_capa) for _ in 1:n_capas_total]

    n_conexiones = n_capas_total - 1

    # Generar pesos con dimensiones correctas: (n_out, n_in)
    pesos = Vector{Matrix{Float64}}(undef, n_conexiones)
    for i in 1:n_conexiones
        n_out = dims[i + 1]
        n_in = dims[i]
        mat = Matrix{Float64}(undef, n_out, n_in)
        for j in eachindex(mat)
            mat[j] = produce!(gen_peso)
        end
        pesos[i] = mat
    end

    # Generar biases con dimensiones correctas: (n_out,)
    biases = Vector{Vector{Float64}}(undef, n_conexiones)
    for i in 1:n_conexiones
        n_out = dims[i + 1]
        vec = Vector{Float64}(undef, n_out)
        for j in eachindex(vec)
            vec[j] = produce!(gen_peso)
        end
        biases[i] = vec
    end

    # Generar activaciones
    activaciones = [produce!(gen_activacion) for _ in 1:n_conexiones]

    RedNeuronal_p3(dims, pesos, biases, activaciones)
end

# ─── Tests de propiedad ──────────────────────────────────────────────────────

@testset "P3: El adaptador preserva todos los atributos de la red" begin

    @check max_examples=100 function p3_adaptador_preserva_atributos(red = gen_red_neuronal)
        resultado = adaptar_red_p3(red, Float64)

        n_conexiones = length(red.capas) - 1

        # (a) Número de capas idéntico
        resultado.capas == red.capas || return false

        # (b) Pesos numéricamente equivalentes
        length(resultado.pesos) == n_conexiones || return false
        for i in 1:n_conexiones
            resultado.pesos[i] ≈ red.pesos[i] || return false
        end

        # (c) Biases numéricamente equivalentes
        length(resultado.biases) == n_conexiones || return false
        for i in 1:n_conexiones
            resultado.biases[i] ≈ red.biases[i] || return false
        end

        # (d) Activaciones idénticas
        resultado.activaciones == red.activaciones || return false

        true
    end
end
