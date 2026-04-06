# Feature: neural-cloud-division, Property 7: Validación de dimensiones de datos
# **Valida: Requisito 6.2**

using Test
using Supposition
using Supposition.Data

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
# Cargamos validar_datos! en aislamiento, sin necesidad del paquete completo.
module _Stubs
    struct MapaDeSoluciones{T<:AbstractFloat}
        data::Vector{T}
    end
    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
    include(joinpath(@__DIR__, "..", "src", "validacion.jl"))
end

const validar_datos! = _Stubs.validar_datos!

# ─── Generadores ──────────────────────────────────────────────────────────────

# Dimensiones positivas para features y outputs
const gen_dim = Data.Integers(1, 20)

# Número de muestras positivo
const gen_n_samples = Data.Integers(1, 50)

# Genera un offset distinto de cero para asegurar n₁ ≠ n₂
const gen_offset_nonzero = Data.Satisfying(
    Data.Integers(-49, 50),
    x -> x != 0
)

# ─── Tests de propiedad ──────────────────────────────────────────────────────

@testset "P7: Validación de dimensiones de datos" begin

    # P7: Para todo par (datos_x, datos_y) con n₁ ≠ n₂, validar_datos! lanza ArgumentError
    @check max_examples=100 function p7_dimensiones_inconsistentes(
        n_features = gen_dim,
        n_outputs  = gen_dim,
        n1         = gen_n_samples,
        offset     = gen_offset_nonzero,
    )
        n2 = n1 + offset
        # Asegurar que n2 sea positivo (válido como dimensión de matriz)
        n2 < 1 && return true  # Caso degenerado, no aplica

        datos_x = zeros(Float64, n_features, n1)
        datos_y = zeros(Float64, n_outputs, n2)

        threw = try
            validar_datos!(datos_x, datos_y)
            false
        catch e
            e isa ArgumentError
        end
        threw
    end
end
