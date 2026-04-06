# motor.jl — Motor_RCND, ejecutar_pipeline, ejecutar_fase_nube, ejecutar_fase_division

using Random: MersenneTwister, AbstractRNG

"""
    Motor_RCND{T<:AbstractFloat, A<:AbstractArray{T}}

Orquestador mutable que mantiene el estado del pipeline y los buffers pre-alocados.

# Campos
- `config`: configuración del pipeline
- `datos_x`: características `(n_features, n_samples)` — column-major
- `datos_y`: etiquetas `(n_outputs, n_samples)` — column-major
- `buffer_activacion`: buffer pre-alocado para forward pass
- `buffer_gradiente`: buffer pre-alocado para backprop
- `rng`: generador de números aleatorios con semilla controlada
- `tiempo_fase_nube`: tiempo de ejecución de la Fase_Nube (segundos)
- `tiempo_fase_division`: tiempo de ejecución de la Fase_Division (segundos)
"""
mutable struct Motor_RCND{T<:AbstractFloat, A<:AbstractArray{T}}
    config::Configuracion_RCND{T}
    datos_x::A
    datos_y::A
    buffer_activacion::A
    buffer_gradiente::A
    rng::AbstractRNG
    tiempo_fase_nube::Float64
    tiempo_fase_division::Float64
end

"""
    Motor_RCND(config::Configuracion_RCND{T}, datos_x::AbstractMatrix{T}, datos_y::AbstractMatrix{T}) where {T}

Construye un `Motor_RCND` validando datos, inicializando RNG y pre-alocando buffers.

# Pasos
1. Valida datos con `validar_datos!` (dimensiones consistentes, no vacíos)
2. Inicializa RNG: `MersenneTwister(semilla)` si la semilla no es `nothing`, sino `MersenneTwister()`
3. Clampea mini-batch: `min(config.tamano_mini_batch, size(datos_x, 2))`
4. Pre-aloca buffers basados en las dimensiones de los datos de entrada
5. Inicializa tiempos a 0.0

# Argumentos
- `config`: configuración del pipeline RCND
- `datos_x`: matriz de características `(n_features, n_samples)`
- `datos_y`: matriz de etiquetas `(n_outputs, n_samples)`

# Errores
- `ArgumentError` si los datos tienen dimensiones inconsistentes o están vacíos
"""
function Motor_RCND(config::Configuracion_RCND{T}, datos_x::AbstractMatrix{T}, datos_y::AbstractMatrix{T}) where {T<:AbstractFloat}
    # 1. Validar datos de entrada
    validar_datos!(datos_x, datos_y)

    # 2. Inicializar RNG con semilla controlada
    rng = if config.semilla !== nothing
        MersenneTwister(config.semilla)
    else
        MersenneTwister()
    end

    # 3. Clampear mini-batch al número de muestras disponibles
    n_samples = size(datos_x, 2)
    tamano_batch_efectivo = min(config.tamano_mini_batch, n_samples)

    # 4. Pre-alocar buffers basados en dimensiones de datos
    #    Usamos las dimensiones de features/outputs como dims_red inicial.
    #    Los buffers se redimensionarán cuando se conozca la arquitectura de red.
    n_features = size(datos_x, 1)
    n_outputs = size(datos_y, 1)
    dims_red = [n_features, n_outputs]

    bufs = crear_buffers(T, dims_red, tamano_batch_efectivo)

    # 5. Construir Motor con tiempos inicializados a 0.0
    A = typeof(datos_x)
    Motor_RCND{T, A}(
        config,
        datos_x,
        datos_y,
        bufs.buffer_activacion,
        bufs.buffer_gradiente,
        rng,
        0.0,
        0.0,
    )
end


# ─── MetricasFase ─────────────────────────────────────────────────────────────

"""
    MetricasFase

NamedTuple que almacena las métricas de una fase del pipeline.

# Campos
- `precision`: precisión alcanzada por la fase (Float64)
- `tiempo`: tiempo de ejecución en segundos (Float64)
- `iteraciones`: número de iteraciones realizadas (Int)
"""
const MetricasFase = NamedTuple{(:precision, :tiempo, :iteraciones), Tuple{Float64, Float64, Int}}

# ─── Callback helper ──────────────────────────────────────────────────────────

"""
    _invocar_callback(config::Configuracion_RCND, args...)

Invoca el callback de fase si está configurado. Si `config.callback_fase` es `nothing`,
no hace nada. Esto permite ejecutar el pipeline con o sin callbacks de forma transparente.

# Argumentos
- `config`: configuración del pipeline con campo `callback_fase`
- `args...`: argumentos a pasar al callback (e.g., `:fase_nube`, `:iniciada`)
"""
function _invocar_callback(config::Configuracion_RCND, args...)
    if config.callback_fase !== nothing
        config.callback_fase(args...)
    end
    nothing
end

# ─── ejecutar_fase_nube ───────────────────────────────────────────────────────

"""
    ejecutar_fase_nube(motor::Motor_RCND{T})::Tuple{RedNeuronal, MetricasFase} where {T}

Ejecuta únicamente la Fase_Nube del pipeline, invocando RandomCloud.jl para buscar
la topología mínima óptima mediante nubes aleatorias (NAS).

# Flujo
1. Invoca callback `(:fase_nube, :iniciada)` si está configurado
2. Llama a `RandomCloud.buscar_topologia` con los parámetros de Fase_Nube
3. Mide el tiempo de ejecución con `@elapsed`
4. Extrae la precisión de la red encontrada
5. Almacena el tiempo en `motor.tiempo_fase_nube`
6. Invoca callback `(:fase_nube, :completada, métricas)` si está configurado
7. Retorna `(red_neuronal, métricas)`

# Argumentos
- `motor`: motor del pipeline con configuración, datos y RNG inicializados

# Retorna
- `Tuple{RedNeuronal, MetricasFase}`: la red encontrada y las métricas de la fase

# Requisitos
- 2.1: Ejecuta la Fase_Nube como primera fase del pipeline
- 2.5: Registra el tiempo de ejecución de la fase
- 7.1: Permite ejecutar la Fase_Nube de forma independiente
- 8.1, 8.2: Invoca callbacks al inicio y fin de la fase con métricas
"""
function ejecutar_fase_nube(motor::Motor_RCND{T})::Tuple{RedNeuronal, MetricasFase} where {T}
    config = motor.config

    # 1. Callback: fase iniciada
    _invocar_callback(config, :fase_nube, :iniciada)

    # 2-3. Ejecutar búsqueda de topología y medir tiempo
    local red_neuronal::RedNeuronal
    tiempo = @elapsed begin
        red_neuronal = RandomCloud.buscar_topologia(
            motor.datos_x, motor.datos_y;
            n_redes=config.n_redes_por_nube,
            activaciones=config.activaciones,
            umbral=config.umbral_precision,
            max_iteraciones=config.max_iteraciones_reduccion,
            rng=motor.rng
        )
    end

    # 4. Extraer precisión de la red encontrada
    precision = Float64(red_neuronal.precision)

    # 5. Almacenar tiempo en el motor
    motor.tiempo_fase_nube = tiempo

    # 6. Construir métricas
    metricas = MetricasFase((precision, tiempo, config.max_iteraciones_reduccion))

    # 7. Callback: fase completada con métricas
    _invocar_callback(config, :fase_nube, :completada, metricas)

    return (red_neuronal, metricas)
end


# ─── ejecutar_fase_division ───────────────────────────────────────────────────

"""
    ejecutar_fase_division(motor::Motor_RCND{T}, red::RedBase{T})::Tuple{MapaDeSoluciones{T}, MetricasFase} where {T}

Ejecuta únicamente la Fase_Division del pipeline, invocando DivisionNeuronal.jl para
descomponer la red en subredes especializadas mediante búsqueda exhaustiva.

# Flujo
1. Invoca callback `(:fase_division, :iniciada)` si está configurado
2. Calcula tamaño efectivo de mini-batch: `min(config.tamano_mini_batch, size(datos_x, 2))`
3. Llama a `DivisionNeuronal.dividir_red` con los parámetros de Fase_Division
4. Mide el tiempo de ejecución con `@elapsed`
5. Almacena el tiempo en `motor.tiempo_fase_division`
6. Construye `MetricasFase` con precisión 0.0 (la precisión real se obtiene del mapa)
7. Invoca callback `(:fase_division, :completada, métricas)` si está configurado
8. Retorna `(mapa_soluciones, métricas)`

# Argumentos
- `motor`: motor del pipeline con configuración, datos y RNG inicializados
- `red`: red base a descomponer (puede provenir de Fase_Nube o ser externa)

# Retorna
- `Tuple{MapaDeSoluciones{T}, MetricasFase}`: el mapa de soluciones y las métricas de la fase

# Requisitos
- 2.1: Ejecuta la Fase_Division como segunda fase del pipeline
- 2.5: Registra el tiempo de ejecución de la fase
- 7.2: Permite ejecutar la Fase_Division de forma independiente
- 7.3: Acepta cualquier RedBase{T} válida (no solo de Fase_Nube)
- 8.1: Invoca callbacks al inicio y fin de la fase
"""
function ejecutar_fase_division(motor::Motor_RCND{T}, red::RedBase{T})::Tuple{MapaDeSoluciones{T}, MetricasFase} where {T}
    config = motor.config

    # 1. Callback: fase iniciada
    _invocar_callback(config, :fase_division, :iniciada)

    # 2. Calcular tamaño efectivo de mini-batch
    effective_batch = min(config.tamano_mini_batch, size(motor.datos_x, 2))

    # 3-4. Ejecutar división neuronal y medir tiempo
    local mapa_soluciones::MapaDeSoluciones{T}
    tiempo = @elapsed begin
        mapa_soluciones = DivisionNeuronal.dividir_red(
            red, motor.datos_x, motor.datos_y;
            umbral=config.umbral_division,
            max_epocas=config.max_epocas,
            tasa_aprendizaje=config.tasa_aprendizaje,
            tamano_mini_batch=effective_batch,
            rng=motor.rng
        )
    end

    # 5. Almacenar tiempo en el motor
    motor.tiempo_fase_division = tiempo

    # 6. Construir métricas
    metricas = MetricasFase((0.0, tiempo, config.max_epocas))

    # 7. Callback: fase completada con métricas
    _invocar_callback(config, :fase_division, :completada, metricas)

    return (mapa_soluciones, metricas)
end


# ─── ejecutar_pipeline ────────────────────────────────────────────────────────

"""
    ejecutar_pipeline(motor::Motor_RCND{T})::Informe_RCND{T} where {T}

Ejecuta el pipeline completo: Fase_Nube → adaptar_red → Fase_Division, y devuelve
un `Informe_RCND{T}` con los resultados combinados.

# Flujo
1. Mide el tiempo total con `@elapsed`
2. Ejecuta `ejecutar_fase_nube(motor)` para obtener la red y métricas
3. Extrae la precisión y verifica si se alcanzó el umbral
4. Si el umbral NO se alcanzó: construye informe parcial con `umbral_alcanzado=false`
5. Si el umbral se alcanzó: adapta la red con `adaptar_red(red, T)` y ejecuta `ejecutar_fase_division`
6. Si la Fase_Division falla: captura el error, preserva resultados de Fase_Nube
7. Transfiere resultados a CPU con `transferir_a_cpu!` antes de construir el informe
8. Construye y retorna `Informe_RCND{T}`

# Argumentos
- `motor`: motor del pipeline con configuración, datos y RNG inicializados

# Retorna
- `Informe_RCND{T}` con resultados de ambas fases

# Requisitos
- 2.1: Ejecuta Fase_Nube y Fase_Division en secuencia
- 2.2: Transfiere la red intermedia automáticamente
- 2.3: Devuelve informe con topología, mapa de soluciones y métricas
- 2.5: Registra tiempos individuales y total
- 6.1: Informe parcial si no se alcanza el umbral
- 6.4: Preserva resultados de Fase_Nube si Fase_Division falla
- 10.7: Transfiere resultados a CPU antes de construir informe
"""
function ejecutar_pipeline(motor::Motor_RCND{T})::Informe_RCND{T} where {T}
    config = motor.config

    # Variables para resultados
    local red_neuronal::RedNeuronal
    local metricas_nube::MetricasFase
    local mapa_soluciones::Union{MapaDeSoluciones{T}, Nothing} = nothing
    local precisiones_subredes::Union{Vector{T}, Nothing} = nothing
    local tiempo_fase_div::Union{Float64, Nothing} = nothing
    local error_div::Union{String, Nothing} = nothing
    local umbral_alcanzado::Bool = false

    # 1. Medir tiempo total
    tiempo_total = @elapsed begin

        # 2. Ejecutar Fase_Nube
        red_neuronal, metricas_nube = ejecutar_fase_nube(motor)

        # 3. Extraer precisión y verificar umbral
        precision_red = T(red_neuronal.precision)
        umbral_alcanzado = precision_red >= config.umbral_precision

        # 4-5. Si el umbral se alcanzó, adaptar red y ejecutar Fase_Division
        if umbral_alcanzado
            # Adaptar red: RedNeuronal → RedBase{T}
            red_base = adaptar_red(red_neuronal, T)

            # Ejecutar Fase_Division con manejo de errores
            try
                mapa, metricas_div = ejecutar_fase_division(motor, red_base)
                mapa_soluciones = mapa
                tiempo_fase_div = metricas_div.tiempo
                # Extraer precisiones de subredes del mapa (data del stub)
                precisiones_subredes = T.(mapa_soluciones.data)
            catch e
                # 6. Fase_Division falló: preservar resultados de Fase_Nube
                error_div = sprint(showerror, e)
                tiempo_fase_div = motor.tiempo_fase_division
            end
        end
    end

    # 7. Transferir resultados a CPU antes de construir informe
    pesos_cpu = transferir_a_cpu!(red_neuronal.pesos)
    biases_cpu = transferir_a_cpu!(red_neuronal.biases)

    # Construir TopologiaOptima desde la RedNeuronal
    topologia = TopologiaOptima{T}(
        copy(red_neuronal.capas),
        [T.(p) for p in pesos_cpu],
        [T.(b) for b in biases_cpu],
        copy(red_neuronal.activaciones),
    )

    # 8. Construir y retornar Informe_RCND
    Informe_RCND{T}(
        topologia,
        T(red_neuronal.precision),
        motor.tiempo_fase_nube,
        mapa_soluciones,
        precisiones_subredes,
        tiempo_fase_div,
        tiempo_total,
        config,
        umbral_alcanzado,
        error_div,
    )
end
