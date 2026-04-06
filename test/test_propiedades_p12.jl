# Feature: neural-cloud-division, Property 12: Selección automática de backend por umbral
# **Valida: Requisitos 10.3, 10.4**

using Test
using Supposition
using Supposition.Data

# ─── Stub de dependencias ─────────────────────────────────────────────────────
module _StubsP12
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

const seleccionar_backend_p12 = _StubsP12.seleccionar_backend
const Configuracion_RCND_p12 = _StubsP12.Configuracion_RCND

# ─── Generadores ──────────────────────────────────────────────────────────────

# Umbral GPU: entero positivo (1 a 100_000)
const gen_umbral_gpu = Data.Integers(1, 100_000)

# Número de parámetros: entero positivo (1 a 100_000)
const gen_n_params = Data.Integers(1, 100_000)

# ─── Test de propiedad ────────────────────────────────────────────────────────

@testset "P12: Selección automática de backend por umbral" begin

    @check max_examples=100 function p12_seleccion_backend_por_umbral(
        umbral_gpu = gen_umbral_gpu,
        n_params = gen_n_params
    )
        config = Configuracion_RCND_p12(backend_computo=:auto, umbral_gpu=umbral_gpu)
        resultado = seleccionar_backend_p12(config, n_params)

        if n_params > umbral_gpu
            resultado === :gpu || return false
        else
            resultado === :cpu || return false
        end

        true
    end
end
