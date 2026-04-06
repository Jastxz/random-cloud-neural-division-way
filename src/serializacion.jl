# serializacion.jl — serializar/deserializar JSON con JSON3.jl
# Requisitos: 5.1, 5.2, 5.3, 5.4

"""
    serializar(informe::Informe_RCND{T})::String where {T}

Convierte un `Informe_RCND{T}` a una cadena JSON válida.
Las matrices se serializan como arrays anidados (Vector de Vectors).
Los campos `Union{..., Nothing}` se serializan como `null` cuando son `nothing`.

# Valida: Requisito 5.1
"""
function serializar(informe::Informe_RCND{T})::String where {T}
    dict = _informe_to_dict(informe)
    JSON3.write(dict)
end

"""
    deserializar(::Type{Informe_RCND{T}}, json_str::String)::Informe_RCND{T} where {T}

Reconstruye un `Informe_RCND{T}` desde una cadena JSON.
Lanza `ErrorDeserializacion` si el JSON es inválido o faltan campos requeridos.

# Valida: Requisitos 5.2, 5.4
"""
function deserializar(::Type{Informe_RCND{T}}, json_str::String)::Informe_RCND{T} where {T}
    local dict
    try
        dict = JSON3.read(json_str)
    catch e
        throw(ErrorDeserializacion("JSON inválido: $(sprint(showerror, e))", nothing))
    end
    _dict_to_informe(T, dict)
end

# ─── Helpers internos: Informe → Dict ─────────────────────────────────────────

"""Convierte un Informe_RCND completo a un Dict serializable."""
function _informe_to_dict(informe::Informe_RCND{T}) where {T}
    d = Dict{String, Any}()
    d["tipo_numerico"] = string(T)
    d["topologia"] = _topologia_to_dict(informe.topologia)
    d["precision_topologia"] = informe.precision_topologia
    d["tiempo_fase_nube"] = informe.tiempo_fase_nube
    d["mapa_soluciones"] = _mapa_to_dict(informe.mapa_soluciones)
    d["precisiones_subredes"] = informe.precisiones_subredes === nothing ? nothing : collect(informe.precisiones_subredes)
    d["tiempo_fase_division"] = informe.tiempo_fase_division
    d["tiempo_total"] = informe.tiempo_total
    d["config_utilizada"] = _config_to_dict(informe.config_utilizada)
    d["umbral_alcanzado"] = informe.umbral_alcanzado
    d["error_fase_division"] = informe.error_fase_division
    return d
end

"""Convierte una TopologiaOptima a Dict, matrices como arrays anidados."""
function _topologia_to_dict(topo::TopologiaOptima{T}) where {T}
    Dict{String, Any}(
        "capas" => collect(topo.capas),
        "pesos" => [_matrix_to_nested(p) for p in topo.pesos],
        "biases" => [collect(b) for b in topo.biases],
        "activaciones" => [string(a) for a in topo.activaciones],
    )
end

"""Convierte una Matrix a un array anidado (Vector de Vectors) por filas."""
function _matrix_to_nested(m::AbstractMatrix)
    [collect(m[i, :]) for i in 1:size(m, 1)]
end

"""Convierte un MapaDeSoluciones a Dict, o nothing si es nothing."""
function _mapa_to_dict(mapa)
    mapa === nothing && return nothing
    Dict{String, Any}("data" => collect(mapa.data))
end

"""Convierte una Configuracion_RCND a Dict (sin callback_fase, no serializable)."""
function _config_to_dict(config::Configuracion_RCND{T}) where {T}
    Dict{String, Any}(
        "tipo_numerico" => string(T),
        "n_redes_por_nube" => config.n_redes_por_nube,
        "activaciones" => [string(a) for a in config.activaciones],
        "umbral_precision" => config.umbral_precision,
        "max_iteraciones_reduccion" => config.max_iteraciones_reduccion,
        "umbral_division" => config.umbral_division,
        "max_epocas" => config.max_epocas,
        "tasa_aprendizaje" => config.tasa_aprendizaje,
        "tamano_mini_batch" => config.tamano_mini_batch,
        "semilla" => config.semilla,
        "verbosidad" => string(config.verbosidad),
        "backend_computo" => string(config.backend_computo),
        "umbral_gpu" => config.umbral_gpu,
    )
end


# ─── Helpers internos: Dict → Informe ─────────────────────────────────────────

"""Reconstruye un Informe_RCND{T} desde un Dict parseado de JSON."""
function _dict_to_informe(::Type{T}, dict) where {T<:AbstractFloat}
    # Reconstruir topología (campo requerido)
    _require_field(dict, "topologia")
    topologia = _dict_to_topologia(T, dict["topologia"])

    # Campos escalares requeridos
    precision_topologia = T(_require_field(dict, "precision_topologia"))
    tiempo_fase_nube = Float64(_require_field(dict, "tiempo_fase_nube"))
    tiempo_total = Float64(_require_field(dict, "tiempo_total"))
    umbral_alcanzado = Bool(_require_field(dict, "umbral_alcanzado"))

    # Campos opcionales (Union{..., Nothing})
    mapa_soluciones = _dict_to_mapa(T, _get_optional(dict, "mapa_soluciones"))
    precisiones_subredes = _get_optional(dict, "precisiones_subredes")
    if precisiones_subredes !== nothing
        precisiones_subredes = T[T(v) for v in precisiones_subredes]
    end
    tiempo_fase_division = _get_optional(dict, "tiempo_fase_division")
    if tiempo_fase_division !== nothing
        tiempo_fase_division = Float64(tiempo_fase_division)
    end
    error_fase_division = _get_optional(dict, "error_fase_division")
    if error_fase_division !== nothing
        error_fase_division = String(error_fase_division)
    end

    # Reconstruir config (campo requerido)
    _require_field(dict, "config_utilizada")
    config_utilizada = _dict_to_config(T, dict["config_utilizada"])

    Informe_RCND{T}(
        topologia,
        precision_topologia,
        tiempo_fase_nube,
        mapa_soluciones,
        precisiones_subredes,
        tiempo_fase_division,
        tiempo_total,
        config_utilizada,
        umbral_alcanzado,
        error_fase_division,
    )
end

"""Reconstruye una TopologiaOptima{T} desde un Dict."""
function _dict_to_topologia(::Type{T}, dict) where {T<:AbstractFloat}
    _require_field(dict, "capas")
    _require_field(dict, "pesos")
    _require_field(dict, "biases")
    _require_field(dict, "activaciones")

    capas = Int[Int(c) for c in dict["capas"]]
    pesos = [_nested_to_matrix(T, p) for p in dict["pesos"]]
    biases = [T[T(v) for v in b] for b in dict["biases"]]
    activaciones = [Symbol(a) for a in dict["activaciones"]]

    TopologiaOptima{T}(capas, pesos, biases, activaciones)
end

"""Reconstruye una Matrix{T} desde un array anidado (Vector de Vectors)."""
function _nested_to_matrix(::Type{T}, nested) where {T<:AbstractFloat}
    n_rows = length(nested)
    if n_rows == 0
        return Matrix{T}(undef, 0, 0)
    end
    n_cols = length(nested[1])
    m = Matrix{T}(undef, n_rows, n_cols)
    for i in 1:n_rows
        row = nested[i]
        if length(row) != n_cols
            throw(ErrorDeserializacion(
                "Dimensiones inconsistentes en matriz: fila $i tiene $(length(row)) columnas, se esperaban $n_cols",
                "topologia.pesos"
            ))
        end
        for j in 1:n_cols
            m[i, j] = T(row[j])
        end
    end
    return m
end

"""Reconstruye un MapaDeSoluciones{T} desde un Dict, o devuelve nothing."""
function _dict_to_mapa(::Type{T}, dict) where {T<:AbstractFloat}
    dict === nothing && return nothing
    _require_field(dict, "data")
    data = T[T(v) for v in dict["data"]]
    MapaDeSoluciones{T}(data)
end

"""Reconstruye una Configuracion_RCND{T} desde un Dict."""
function _dict_to_config(::Type{T}, dict) where {T<:AbstractFloat}
    _require_field(dict, "n_redes_por_nube")
    _require_field(dict, "activaciones")
    _require_field(dict, "umbral_precision")
    _require_field(dict, "max_iteraciones_reduccion")
    _require_field(dict, "umbral_division")
    _require_field(dict, "max_epocas")
    _require_field(dict, "tasa_aprendizaje")
    _require_field(dict, "tamano_mini_batch")
    _require_field(dict, "verbosidad")
    _require_field(dict, "backend_computo")
    _require_field(dict, "umbral_gpu")

    Configuracion_RCND{T}(
        n_redes_por_nube = Int(dict["n_redes_por_nube"]),
        activaciones = Symbol[Symbol(a) for a in dict["activaciones"]],
        umbral_precision = T(dict["umbral_precision"]),
        max_iteraciones_reduccion = Int(dict["max_iteraciones_reduccion"]),
        umbral_division = T(dict["umbral_division"]),
        max_epocas = Int(dict["max_epocas"]),
        tasa_aprendizaje = T(dict["tasa_aprendizaje"]),
        tamano_mini_batch = Int(dict["tamano_mini_batch"]),
        semilla = _get_optional(dict, "semilla") === nothing ? nothing : Int(dict["semilla"]),
        verbosidad = Symbol(dict["verbosidad"]),
        backend_computo = Symbol(dict["backend_computo"]),
        umbral_gpu = Int(dict["umbral_gpu"]),
        callback_fase = nothing,  # Callbacks no son serializables
    )
end

# ─── Utilidades de validación ─────────────────────────────────────────────────

"""Verifica que un campo existe en el Dict. Lanza ErrorDeserializacion si falta."""
function _require_field(dict, campo::String)
    if !haskey(dict, campo)
        throw(ErrorDeserializacion("Campo requerido faltante: '$campo'", campo))
    end
    return dict[campo]
end

"""Obtiene un campo opcional del Dict, devolviendo nothing si no existe o es null."""
function _get_optional(dict, campo::String)
    if !haskey(dict, campo)
        return nothing
    end
    val = dict[campo]
    val === nothing && return nothing
    return val
end
