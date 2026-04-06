using Test

@testset "RandomCloudNeuralDivision Tests" begin
    # ── Tests unitarios ──────────────────────────────────────────────────────
    include("test_tipos.jl")
    include("test_validacion.jl")
    include("test_buffers.jl")
    include("test_adaptador.jl")
    include("test_backend.jl")
    include("test_motor.jl")
    include("test_informe.jl")
    include("test_serializacion.jl")
    include("test_pipeline.jl")

    # ── Tests de propiedades (PBT) — agrupados ───────────────────────────────
    include("test_propiedades.jl")
end
