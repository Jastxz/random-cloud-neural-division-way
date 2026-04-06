# test_motor.jl — Tests unitarios del constructor Motor_RCND, ejecutar_fase_nube y ejecutar_fase_division
# Valida: Requisitos 2.1, 2.4, 2.5, 6.2, 6.3, 7.1, 7.2, 7.3, 8.1, 8.2, 9.2, 10.3, 10.4

using Test
using Random

# ─── Stub de dependencias ─────────────────────────────────────────────────────
module _StubsMotor
    using Random

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

    # Stub de RedBase (DivisionNeuronal.jl)
    struct RedBase{T<:AbstractFloat}
        capas::Vector{Int}
        pesos::Vector{Matrix{T}}
        biases::Vector{Vector{T}}
        activaciones::Vector{Symbol}
    end

    # Stub de RedNeuronal (RandomCloud.jl)
    struct RedNeuronal
        capas::Vector{Int}
        pesos::Vector{<:AbstractMatrix}
        biases::Vector{<:AbstractVector}
        activaciones::Vector{Symbol}
        precision::Float64
    end

    # Stub de RandomCloud module
    module RandomCloud
        using Random: AbstractRNG
        import ..RedNeuronal

        """Stub de buscar_topologia que retorna una red simple determinista."""
        function buscar_topologia(datos_x, datos_y;
                n_redes::Int=100,
                activaciones::Vector{Symbol}=[:sigmoid],
                umbral::Real=0.95,
                max_iteraciones::Int=10,
                rng::AbstractRNG=Random.default_rng())
            n_in = size(datos_x, 1)
            n_out = size(datos_y, 1)
            # Generar pesos deterministas usando el rng proporcionado
            pesos = [randn(rng, n_out, n_in)]
            biases = [randn(rng, n_out)]
            RedNeuronal(
                [n_in, n_out],
                pesos,
                biases,
                [activaciones[1]],
                min(Float64(umbral), 0.98)  # Simula precisión cercana al umbral
            )
        end
    end

    # Stub de DivisionNeuronal module
    module DivisionNeuronal
        using Random: AbstractRNG
        import ..RedBase, ..MapaDeSoluciones

        # Flag global para simular errores en dividir_red
        const _simular_error = Ref(false)

        """Stub de dividir_red que retorna un MapaDeSoluciones simple determinista."""
        function dividir_red(red::RedBase{T}, datos_x, datos_y;
                umbral::Real=0.5,
                max_epocas::Int=100,
                tasa_aprendizaje::Real=0.001,
                tamano_mini_batch::Int=32,
                rng::AbstractRNG=Random.default_rng()) where {T}
            if _simular_error[]
                error("Error simulado en Fase_Division: descomposición fallida")
            end
            # Retorna un MapaDeSoluciones simple con datos deterministas basados en rng
            n_out = size(datos_y, 1)
            data = T[randn(rng, T) for _ in 1:n_out]
            MapaDeSoluciones{T}(data)
        end
    end

    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
    include(joinpath(@__DIR__, "..", "src", "validacion.jl"))
    include(joinpath(@__DIR__, "..", "src", "buffers.jl"))
    include(joinpath(@__DIR__, "..", "src", "backend.jl"))
    include(joinpath(@__DIR__, "..", "src", "adaptador.jl"))
    include(joinpath(@__DIR__, "..", "src", "motor.jl"))
end

const Motor_RCND_T = _StubsMotor.Motor_RCND
const Config_T = _StubsMotor.Configuracion_RCND
const MetricasFase_T = _StubsMotor.MetricasFase
const ejecutar_fase_nube_T = _StubsMotor.ejecutar_fase_nube
const ejecutar_fase_division_T = _StubsMotor.ejecutar_fase_division
const ejecutar_pipeline_T = _StubsMotor.ejecutar_pipeline
const RedBase_T = _StubsMotor.RedBase
const MapaDeSoluciones_T = _StubsMotor.MapaDeSoluciones
const Informe_RCND_T = _StubsMotor.Informe_RCND
const TopologiaOptima_T = _StubsMotor.TopologiaOptima

@testset "Motor_RCND — Constructor" begin

    @testset "Construcción básica con datos válidos" begin
        config = Config_T{Float64}()
        datos_x = randn(4, 20)   # 4 features, 20 samples
        datos_y = randn(2, 20)   # 2 outputs, 20 samples

        motor = Motor_RCND_T(config, datos_x, datos_y)

        @test motor.config === config
        @test motor.datos_x === datos_x
        @test motor.datos_y === datos_y
        @test motor.tiempo_fase_nube == 0.0
        @test motor.tiempo_fase_division == 0.0
        @test motor.rng isa Random.AbstractRNG
        @test size(motor.buffer_activacion, 2) == min(config.tamano_mini_batch, 20)
        @test size(motor.buffer_gradiente, 2) == min(config.tamano_mini_batch, 20)
    end

    @testset "RNG con semilla fija produce estado reproducible" begin
        config = Config_T{Float64}(semilla=42)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)

        motor1 = Motor_RCND_T(config, datos_x, datos_y)
        motor2 = Motor_RCND_T(config, datos_x, datos_y)

        # Ambos RNG deben producir la misma secuencia
        @test rand(motor1.rng) == rand(motor2.rng)
    end

    @testset "RNG sin semilla (nothing) inicializa correctamente" begin
        config = Config_T{Float64}(semilla=nothing)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)

        motor = Motor_RCND_T(config, datos_x, datos_y)
        @test motor.rng isa Random.MersenneTwister
    end

    @testset "Clampeo de mini-batch cuando batch > n_samples" begin
        config = Config_T{Float64}(tamano_mini_batch=100)
        datos_x = randn(3, 5)   # Solo 5 muestras
        datos_y = randn(1, 5)

        motor = Motor_RCND_T(config, datos_x, datos_y)

        # Buffers deben tener n_samples=5 columnas (clampeado)
        @test size(motor.buffer_activacion, 2) == 5
        @test size(motor.buffer_gradiente, 2) == 5
    end

    @testset "Clampeo de mini-batch cuando batch <= n_samples" begin
        config = Config_T{Float64}(tamano_mini_batch=8)
        datos_x = randn(3, 20)
        datos_y = randn(1, 20)

        motor = Motor_RCND_T(config, datos_x, datos_y)

        # Buffers deben tener n_samples=8 columnas (batch size)
        @test size(motor.buffer_activacion, 2) == 8
        @test size(motor.buffer_gradiente, 2) == 8
    end

    @testset "Lanza ArgumentError con dimensiones inconsistentes" begin
        config = Config_T{Float64}()
        datos_x = randn(3, 10)
        datos_y = randn(1, 7)   # Diferente número de muestras

        @test_throws ArgumentError Motor_RCND_T(config, datos_x, datos_y)
    end

    @testset "Lanza ArgumentError con datos vacíos" begin
        config = Config_T{Float64}()
        datos_x = zeros(3, 0)
        datos_y = zeros(1, 0)

        @test_throws ArgumentError Motor_RCND_T(config, datos_x, datos_y)
    end

    @testset "Funciona con Float32" begin
        config = Config_T{Float32}()
        datos_x = randn(Float32, 4, 15)
        datos_y = randn(Float32, 2, 15)

        motor = Motor_RCND_T(config, datos_x, datos_y)

        @test eltype(motor.buffer_activacion) == Float32
        @test eltype(motor.buffer_gradiente) == Float32
    end

    @testset "Buffer dimensiones basadas en datos" begin
        config = Config_T{Float64}(tamano_mini_batch=10)
        datos_x = randn(8, 20)   # 8 features
        datos_y = randn(3, 20)   # 3 outputs

        motor = Motor_RCND_T(config, datos_x, datos_y)

        # max_dim = max(8, 3) = 8
        @test size(motor.buffer_activacion, 1) == 8
        @test size(motor.buffer_gradiente, 1) == 8
        @test size(motor.buffer_activacion, 2) == 10
    end
end

# ─── Tests de ejecutar_fase_nube ──────────────────────────────────────────────
# Valida: Requisitos 2.1, 2.5, 7.1, 8.1, 8.2

@testset "ejecutar_fase_nube" begin

    @testset "Retorna RedNeuronal y MetricasFase válidas" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.9)
        datos_x = randn(4, 20)
        datos_y = randn(2, 20)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        red, metricas = ejecutar_fase_nube_T(motor)

        # RedNeuronal debe tener estructura válida
        @test red isa _StubsMotor.RedNeuronal
        @test length(red.capas) >= 2
        @test length(red.pesos) == length(red.capas) - 1
        @test length(red.biases) == length(red.capas) - 1

        # MetricasFase debe tener los campos correctos
        @test metricas isa MetricasFase_T
        @test metricas.precision >= 0.0
        @test metricas.tiempo >= 0.0
        @test metricas.iteraciones == config.max_iteraciones_reduccion
    end

    @testset "Mide tiempo de ejecución y lo almacena en motor" begin
        config = Config_T{Float64}(semilla=123)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        @test motor.tiempo_fase_nube == 0.0

        _, metricas = ejecutar_fase_nube_T(motor)

        # El tiempo debe haberse actualizado
        @test motor.tiempo_fase_nube >= 0.0
        @test motor.tiempo_fase_nube == metricas.tiempo
    end

    @testset "Usa motor.rng para reproducibilidad (Req 2.4)" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.85)
        datos_x = randn(3, 15)
        datos_y = randn(1, 15)

        motor1 = Motor_RCND_T(config, datos_x, datos_y)
        motor2 = Motor_RCND_T(config, datos_x, datos_y)

        red1, _ = ejecutar_fase_nube_T(motor1)
        red2, _ = ejecutar_fase_nube_T(motor2)

        # Misma semilla → mismos pesos
        @test red1.pesos[1] == red2.pesos[1]
        @test red1.biases[1] == red2.biases[1]
        @test red1.precision == red2.precision
    end

    @testset "Invoca callbacks al inicio y fin (Req 8.1, 8.2)" begin
        registro = []
        callback = (args...) -> push!(registro, args)

        config = Config_T{Float64}(semilla=42, callback_fase=callback)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        ejecutar_fase_nube_T(motor)

        # Debe haber exactamente 2 invocaciones
        @test length(registro) == 2

        # Primera: (:fase_nube, :iniciada)
        @test registro[1] == (:fase_nube, :iniciada)

        # Segunda: (:fase_nube, :completada, métricas)
        @test registro[2][1] == :fase_nube
        @test registro[2][2] == :completada
        @test registro[2][3] isa MetricasFase_T
        @test registro[2][3].precision >= 0.0
        @test registro[2][3].tiempo >= 0.0
    end

    @testset "Funciona sin callback (Req 8.3)" begin
        config = Config_T{Float64}(semilla=42, callback_fase=nothing)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        # No debe lanzar error
        red, metricas = ejecutar_fase_nube_T(motor)
        @test red isa _StubsMotor.RedNeuronal
        @test metricas isa MetricasFase_T
    end

    @testset "Precisión de métricas coincide con la de la red" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.75)
        datos_x = randn(5, 30)
        datos_y = randn(2, 30)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        red, metricas = ejecutar_fase_nube_T(motor)

        @test metricas.precision == Float64(red.precision)
    end

    @testset "Funciona con Float32" begin
        config = Config_T{Float32}(semilla=42)
        datos_x = randn(Float32, 3, 10)
        datos_y = randn(Float32, 1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        red, metricas = ejecutar_fase_nube_T(motor)

        @test red isa _StubsMotor.RedNeuronal
        @test metricas isa MetricasFase_T
        @test metricas.tiempo >= 0.0
    end
end


# ─── Tests de ejecutar_fase_division ──────────────────────────────────────────
# Valida: Requisitos 2.1, 2.5, 7.2, 7.3, 8.1

# Helper: crea una RedBase{T} simple para tests
function _crear_red_base(::Type{T}, n_in::Int, n_out::Int; rng=Random.default_rng()) where {T}
    pesos = [randn(rng, T, n_out, n_in)]
    biases = [randn(rng, T, n_out)]
    RedBase_T{T}([n_in, n_out], pesos, biases, [:sigmoid])
end

@testset "ejecutar_fase_division" begin

    @testset "Retorna MapaDeSoluciones y MetricasFase válidas (Req 7.2)" begin
        config = Config_T{Float64}(semilla=42)
        datos_x = randn(4, 20)
        datos_y = randn(2, 20)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        red = _crear_red_base(Float64, 4, 2)
        mapa, metricas = ejecutar_fase_division_T(motor, red)

        @test mapa isa MapaDeSoluciones_T{Float64}
        @test metricas isa MetricasFase_T
        @test metricas.tiempo >= 0.0
        @test metricas.iteraciones == config.max_epocas
    end

    @testset "Mide tiempo de ejecución y lo almacena en motor (Req 2.5)" begin
        config = Config_T{Float64}(semilla=123)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        @test motor.tiempo_fase_division == 0.0

        red = _crear_red_base(Float64, 3, 1)
        _, metricas = ejecutar_fase_division_T(motor, red)

        @test motor.tiempo_fase_division >= 0.0
        @test motor.tiempo_fase_division == metricas.tiempo
    end

    @testset "Acepta RedBase externa no proveniente de Fase_Nube (Req 7.3)" begin
        config = Config_T{Float64}(semilla=42)
        datos_x = randn(5, 15)
        datos_y = randn(3, 15)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        # Crear una RedBase externa con arquitectura diferente a los datos
        red_externa = RedBase_T{Float64}(
            [5, 10, 3],
            [randn(10, 5), randn(3, 10)],
            [randn(10), randn(3)],
            [:relu, :sigmoid]
        )

        mapa, metricas = ejecutar_fase_division_T(motor, red_externa)

        @test mapa isa MapaDeSoluciones_T{Float64}
        @test metricas isa MetricasFase_T
    end

    @testset "Invoca callbacks al inicio y fin (Req 8.1)" begin
        registro = []
        callback = (args...) -> push!(registro, args)

        config = Config_T{Float64}(semilla=42, callback_fase=callback)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        red = _crear_red_base(Float64, 3, 1)
        ejecutar_fase_division_T(motor, red)

        # Debe haber exactamente 2 invocaciones
        @test length(registro) == 2

        # Primera: (:fase_division, :iniciada)
        @test registro[1] == (:fase_division, :iniciada)

        # Segunda: (:fase_division, :completada, métricas)
        @test registro[2][1] == :fase_division
        @test registro[2][2] == :completada
        @test registro[2][3] isa MetricasFase_T
        @test registro[2][3].tiempo >= 0.0
    end

    @testset "Funciona sin callback" begin
        config = Config_T{Float64}(semilla=42, callback_fase=nothing)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        red = _crear_red_base(Float64, 3, 1)
        mapa, metricas = ejecutar_fase_division_T(motor, red)

        @test mapa isa MapaDeSoluciones_T{Float64}
        @test metricas isa MetricasFase_T
    end

    @testset "Usa motor.rng para reproducibilidad (Req 2.4)" begin
        config = Config_T{Float64}(semilla=42)
        datos_x = randn(3, 15)
        datos_y = randn(1, 15)

        red = _crear_red_base(Float64, 3, 1)

        motor1 = Motor_RCND_T(config, datos_x, datos_y)
        motor2 = Motor_RCND_T(config, datos_x, datos_y)

        mapa1, _ = ejecutar_fase_division_T(motor1, red)
        mapa2, _ = ejecutar_fase_division_T(motor2, red)

        # Misma semilla → mismos resultados
        @test mapa1.data == mapa2.data
    end

    @testset "Funciona con Float32" begin
        config = Config_T{Float32}(semilla=42)
        datos_x = randn(Float32, 3, 10)
        datos_y = randn(Float32, 1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        red = _crear_red_base(Float32, 3, 1)
        mapa, metricas = ejecutar_fase_division_T(motor, red)

        @test mapa isa MapaDeSoluciones_T{Float32}
        @test metricas isa MetricasFase_T
        @test metricas.tiempo >= 0.0
    end

    @testset "Effective batch clampeado a n_samples (Req 9.2)" begin
        # Config con batch grande, datos pequeños
        config = Config_T{Float64}(semilla=42, tamano_mini_batch=100)
        datos_x = randn(3, 5)   # Solo 5 muestras
        datos_y = randn(1, 5)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        red = _crear_red_base(Float64, 3, 1)
        # Should not error — effective batch is clamped to 5
        mapa, metricas = ejecutar_fase_division_T(motor, red)

        @test mapa isa MapaDeSoluciones_T{Float64}
        @test metricas isa MetricasFase_T
    end
end


# ─── Tests de ejecutar_pipeline ───────────────────────────────────────────────
# Valida: Requisitos 2.1, 2.2, 2.3, 2.5, 6.1, 6.4, 10.7

@testset "ejecutar_pipeline" begin

    @testset "Pipeline completo retorna Informe_RCND válido (Req 2.1, 2.3)" begin
        # Umbral bajo para que el stub lo alcance (stub retorna min(umbral, 0.98))
        config = Config_T{Float64}(semilla=42, umbral_precision=0.5)
        datos_x = randn(4, 20)
        datos_y = randn(2, 20)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        informe = ejecutar_pipeline_T(motor)

        @test informe isa Informe_RCND_T{Float64}
        @test informe.topologia isa TopologiaOptima_T{Float64}
        @test informe.precision_topologia >= 0.0
        @test informe.tiempo_fase_nube >= 0.0
        @test informe.tiempo_total >= 0.0
        @test informe.config_utilizada === config
        @test informe.umbral_alcanzado == true
        @test informe.mapa_soluciones !== nothing
        @test informe.tiempo_fase_division !== nothing
        @test informe.error_fase_division === nothing
    end

    @testset "Registra tiempos individuales y total (Req 2.5)" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.5)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        informe = ejecutar_pipeline_T(motor)

        @test informe.tiempo_fase_nube >= 0.0
        @test informe.tiempo_fase_division !== nothing
        @test informe.tiempo_fase_division >= 0.0
        @test informe.tiempo_total >= informe.tiempo_fase_nube + informe.tiempo_fase_division
    end

    @testset "Topología construida correctamente desde RedNeuronal (Req 2.2)" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.5)
        datos_x = randn(4, 20)
        datos_y = randn(2, 20)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        informe = ejecutar_pipeline_T(motor)

        topo = informe.topologia
        @test length(topo.capas) >= 2
        @test length(topo.pesos) == length(topo.capas) - 1
        @test length(topo.biases) == length(topo.capas) - 1
        @test length(topo.activaciones) == length(topo.capas) - 1
        # Pesos deben ser CPU arrays (Matrix{Float64})
        @test topo.pesos[1] isa Matrix{Float64}
        @test topo.biases[1] isa Vector{Float64}
    end

    @testset "Informe parcial cuando umbral no alcanzado (Req 6.1)" begin
        # Umbral muy alto: stub retorna min(umbral, 0.98), así que 0.99 > 0.98
        config = Config_T{Float64}(semilla=42, umbral_precision=0.99)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        informe = ejecutar_pipeline_T(motor)

        @test informe.umbral_alcanzado == false
        @test informe.mapa_soluciones === nothing
        @test informe.precisiones_subredes === nothing
        @test informe.tiempo_fase_division === nothing
        @test informe.error_fase_division === nothing
        # Topología debe estar presente (mejor encontrada)
        @test informe.topologia isa TopologiaOptima_T{Float64}
        @test informe.precision_topologia >= 0.0
        @test informe.tiempo_fase_nube >= 0.0
        @test informe.tiempo_total >= 0.0
    end

    @testset "Determinismo con semilla fija (Req 2.4)" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.5)
        datos_x = randn(3, 15)
        datos_y = randn(1, 15)

        motor1 = Motor_RCND_T(config, datos_x, datos_y)
        motor2 = Motor_RCND_T(config, datos_x, datos_y)

        informe1 = ejecutar_pipeline_T(motor1)
        informe2 = ejecutar_pipeline_T(motor2)

        @test informe1.topologia.pesos == informe2.topologia.pesos
        @test informe1.topologia.biases == informe2.topologia.biases
        @test informe1.precision_topologia == informe2.precision_topologia
        @test informe1.umbral_alcanzado == informe2.umbral_alcanzado
    end

    @testset "Funciona con Float32" begin
        config = Config_T{Float32}(semilla=42, umbral_precision=Float32(0.5))
        datos_x = randn(Float32, 3, 10)
        datos_y = randn(Float32, 1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        informe = ejecutar_pipeline_T(motor)

        @test informe isa Informe_RCND_T{Float32}
        @test informe.topologia.pesos[1] isa Matrix{Float32}
        @test informe.topologia.biases[1] isa Vector{Float32}
        @test informe.precision_topologia isa Float32
    end

    @testset "Config utilizada se preserva en informe (Req 4.4)" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.5, max_epocas=200)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        informe = ejecutar_pipeline_T(motor)

        @test informe.config_utilizada === config
        @test informe.config_utilizada.max_epocas == 200
    end

    @testset "Precisiones de subredes presentes cuando pipeline completo" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.5)
        datos_x = randn(3, 10)
        datos_y = randn(2, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        informe = ejecutar_pipeline_T(motor)

        if informe.umbral_alcanzado
            @test informe.precisiones_subredes !== nothing
            @test informe.precisiones_subredes isa Vector{Float64}
        end
    end

    @testset "Error en Fase_Division preserva resultados de Fase_Nube (Req 6.4)" begin
        config = Config_T{Float64}(semilla=42, umbral_precision=0.5)
        datos_x = randn(3, 10)
        datos_y = randn(1, 10)
        motor = Motor_RCND_T(config, datos_x, datos_y)

        # Activar simulación de error en Fase_Division
        _StubsMotor.DivisionNeuronal._simular_error[] = true
        try
            informe = ejecutar_pipeline_T(motor)

            # Fase_Nube results preserved
            @test informe.topologia isa TopologiaOptima_T{Float64}
            @test informe.precision_topologia >= 0.0
            @test informe.tiempo_fase_nube >= 0.0
            @test informe.umbral_alcanzado == true

            # Fase_Division error captured
            @test informe.error_fase_division !== nothing
            @test occursin("Error simulado", informe.error_fase_division)
            @test informe.mapa_soluciones === nothing
            @test informe.precisiones_subredes === nothing
            @test informe.tiempo_total >= 0.0
        finally
            _StubsMotor.DivisionNeuronal._simular_error[] = false
        end
    end
end
