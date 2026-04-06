# stubs.jl — Carga de los paquetes reales RandomCloud.jl y DivisionNeuronal.jl
#
# Carga ambos paquetes desde sus directorios fuente y expone una API
# unificada para los benchmarks.

module RCND
    using Random
    using Printf

    # ─── Cargar RandomCloud.jl ────────────────────────────────────────────────
    include(joinpath(@__DIR__, "..", "src", "RandomCloud", "RandomCloud.jl"))
    using .RandomCloud

    # ─── Cargar DivisionNeuronal.jl ───────────────────────────────────────────
    include(joinpath(@__DIR__, "..", "src", "NeuralDivision", "DivisionNeuronal.jl"))
    using .DivisionNeuronal

    # ─── API unificada para benchmarks ────────────────────────────────────────

    """
    Resultado unificado de un benchmark RCND.
    """
    struct InformeBenchmark
        # Fase Nube
        red_encontrada::Bool
        precision_nube::Float64
        topologia_final::Union{Vector{Int}, Nothing}
        n_parametros_nube::Int
        tiempo_nube_ms::Float64
        exitoso_nube::Bool

        # Fase División
        division_ejecutada::Bool
        precision_global_div::Float64
        n_soluciones_parciales::Int
        tiempo_division_ms::Float64

        # Global
        tiempo_total_ms::Float64
    end

    """
        ejecutar_benchmark_rcnd(datos_x, datos_y; kwargs...) -> InformeBenchmark

    Ejecuta el pipeline completo RCND con los paquetes reales.

    datos_x: (n_features, n_samples) — column-major
    datos_y: (n_outputs, n_samples) — column-major

    Kwargs para Fase Nube:
    - tamano_nube: número de redes por nube (default 10)
    - topologia_inicial: topología inicial [entrada, ocultas..., salida]
    - umbral_acierto: umbral de precisión (default 0.9)
    - neuronas_eliminar: neuronas a eliminar por reducción (default 1)
    - epocas_refinamiento: épocas de entrenamiento post-NAS (default 500)
    - tasa_aprendizaje_nube: learning rate para refinamiento (default 0.1)
    - semilla: semilla aleatoria (default 42)
    - activacion: función de activación (default :sigmoid)
    - batch_size_nube: tamaño de mini-batch para refinamiento (default 0 = sample-by-sample)

    Kwargs para Fase División:
    - umbral_division: umbral de acierto para división (default 0.4)
    - epochs_division: épocas de entrenamiento de subredes (default 500)
    - lr_division: learning rate para Adam (default 0.001)
    - batch_size_division: tamaño de mini-batch (default 32)
    - paciencia_division: early stopping patience (default 50)
    """
    function ejecutar_benchmark_rcnd(datos_x::Matrix{Float64}, datos_y::Matrix{Float64};
            tamano_nube::Int = 10,
            topologia_inicial::Union{Vector{Int}, Nothing} = nothing,
            umbral_acierto::Float64 = 0.9,
            neuronas_eliminar::Int = 1,
            epocas_refinamiento::Int = 500,
            tasa_aprendizaje_nube::Float64 = 0.1,
            semilla::Int = 42,
            activacion::Symbol = :sigmoid,
            batch_size_nube::Int = 0,
            umbral_division::Float64 = 0.4,
            epochs_division::Int = 500,
            lr_division::Float64 = 0.001,
            batch_size_division::Int = 32,
            paciencia_division::Int = 50)

        n_features = size(datos_x, 1)
        n_outputs = size(datos_y, 1)
        n_samples = size(datos_x, 2)

        # Topología inicial por defecto: [entrada, 2*entrada, salida]
        if topologia_inicial === nothing
            n_hidden = max(2, min(n_features * 2, 20))
            topologia_inicial = [n_features, n_hidden, n_outputs]
        end

        t_total_inicio = time_ns()

        # ═══ FASE NUBE ═══════════════════════════════════════════════════════
        config_nube = ConfiguracionNube(
            tamano_nube = tamano_nube,
            topologia_inicial = topologia_inicial,
            umbral_acierto = umbral_acierto,
            neuronas_eliminar = neuronas_eliminar,
            epocas_refinamiento = epocas_refinamiento,
            tasa_aprendizaje = tasa_aprendizaje_nube,
            semilla = semilla,
            activacion = activacion,
            batch_size = batch_size_nube,
        )

        motor_nube = MotorNube(config_nube, datos_x, datos_y)
        informe_nube = ejecutar(motor_nube)

        n_params_nube = 0
        if informe_nube.mejor_red !== nothing
            for p in informe_nube.mejor_red.pesos
                n_params_nube += length(p)
            end
            for b in informe_nube.mejor_red.biases
                n_params_nube += length(b)
            end
        end

        # ═══ FASE DIVISIÓN ═══════════════════════════════════════════════════
        division_ejecutada = false
        precision_global_div = 0.0
        n_soluciones_parciales = 0
        tiempo_division_ms = 0.0

        if informe_nube.exitoso && informe_nube.mejor_red !== nothing
            red_rc = informe_nube.mejor_red
            topo = red_rc.topologia

            # Convertir RedNeuronal (column-major) → RedBase (row-major)
            # RandomCloud: pesos[i] es (n_out, n_in) — column-major
            # DivisionNeuronal: pesos[i] es (n_in, n_out) — row-major
            pesos_div = [Matrix{Float64}(p') for p in red_rc.pesos]
            biases_div = [copy(b) for b in red_rc.biases]

            red_base = RedBase{Float64}(
                pesos_div,
                biases_div,
                topo[1],      # n_entradas
                topo[end],    # n_salidas
            )

            # Preparar datos para DivisionNeuronal (row-major: samples × features)
            datos_val = (
                entradas = Matrix{Float64}(datos_x'),   # (samples, features)
                salidas = Matrix{Float64}(datos_y'),     # (samples, outputs)
            )

            config_div = ConfiguracionDivision{Float64}(umbral_division)

            t_div_inicio = time_ns()
            try
                mapa = ejecutar_division(
                    red_base, datos_val, config_div;
                    epochs = epochs_division,
                    lr = lr_division,
                    batch_size = batch_size_division,
                    paciencia = paciencia_division,
                )
                t_div_fin = time_ns()
                tiempo_division_ms = (t_div_fin - t_div_inicio) / 1_000_000.0
                division_ejecutada = true

                # Extraer métricas del mapa
                if mapa.global_.subconfiguracion !== nothing
                    precision_global_div = Float64(mapa.global_.precision)
                end
                n_soluciones_parciales = count(
                    e -> e.subconfiguracion !== nothing,
                    values(mapa.parciales)
                )
            catch e
                t_div_fin = time_ns()
                tiempo_division_ms = (t_div_fin - t_div_inicio) / 1_000_000.0
                @warn "Error en Fase División" exception=e
            end
        end

        t_total_fin = time_ns()
        tiempo_total_ms = (t_total_fin - t_total_inicio) / 1_000_000.0

        InformeBenchmark(
            informe_nube.mejor_red !== nothing,
            informe_nube.precision,
            informe_nube.topologia_final,
            n_params_nube,
            informe_nube.tiempo_ejecucion_ms,
            informe_nube.exitoso,
            division_ejecutada,
            precision_global_div,
            n_soluciones_parciales,
            tiempo_division_ms,
            tiempo_total_ms,
        )
    end
end
