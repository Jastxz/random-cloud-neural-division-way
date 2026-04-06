# Feature: neural-cloud-division, Property 2: Rechazo de parámetros inválidos
# **Valida: Requisitos 1.5, 1.6, 9.3**

using Test
using Supposition
using Supposition.Data

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
# MapaDeSoluciones es un tipo de DivisionNeuronal.jl (paquete ficticio).
# Lo definimos aquí como stub para poder cargar tipos.jl en aislamiento.
module _Stubs
    struct MapaDeSoluciones{T<:AbstractFloat}
        data::Vector{T}
    end
    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
end

const Configuracion_RCND = _Stubs.Configuracion_RCND

# ─── Generadores ──────────────────────────────────────────────────────────────

# Genera Float64 finitos fuera del rango [0, 1]: negativos o > 1
const gen_umbral_invalido = Data.OneOf(
    Data.Satisfying(Data.Floats{Float64}(), x -> isfinite(x) && x < 0.0),
    Data.Satisfying(Data.Floats{Float64}(), x -> isfinite(x) && x > 1.0),
)

# Genera enteros ≤ 0 para n_redes_por_nube
const gen_n_redes_invalido = Data.Integers(typemin(Int) ÷ 2, 0)

# Genera enteros ≤ 0 para tamano_mini_batch
const gen_batch_invalido = Data.Integers(typemin(Int) ÷ 2, 0)

# ─── Tests de propiedad ──────────────────────────────────────────────────────

@testset "P2: Rechazo de parámetros inválidos" begin

    # P2a: umbral_precision fuera de [0, 1] → ArgumentError
    @check max_examples=100 function p2a_umbral_invalido(u = gen_umbral_invalido)
        threw = try
            Configuracion_RCND{Float64}(umbral_precision=u)
            false
        catch e
            e isa ArgumentError
        end
        threw
    end

    # P2b: n_redes_por_nube < 1 → ArgumentError
    @check max_examples=100 function p2b_n_redes_invalido(n = gen_n_redes_invalido)
        threw = try
            Configuracion_RCND{Float64}(n_redes_por_nube=n)
            false
        catch e
            e isa ArgumentError
        end
        threw
    end

    # P2c: tamano_mini_batch < 1 → ArgumentError
    @check max_examples=100 function p2c_batch_invalido(b = gen_batch_invalido)
        threw = try
            Configuracion_RCND{Float64}(tamano_mini_batch=b)
            false
        catch e
            e isa ArgumentError
        end
        threw
    end
end
