# adaptador.jl — Conversión RedNeuronal → RedBase, conteo de parámetros

"""
    validar_arquitectura(red::RedNeuronal)

Valida que la arquitectura de una RedNeuronal es compatible con la Fase_Division.
Verifica consistencia entre dimensiones de capas, pesos y biases.

# Errores
- `ErrorAdaptador` si las dimensiones de pesos o biases no coinciden con las capas.
"""
function validar_arquitectura(red::RedNeuronal)
    capas = red.capas
    pesos = red.pesos
    biases = red.biases
    activaciones = red.activaciones

    n_capas = length(capas)

    # Necesitamos al menos 2 capas (entrada + salida)
    if n_capas < 2
        throw(ErrorAdaptador(
            "La red debe tener al menos 2 capas (entrada y salida), tiene: $n_capas",
            0
        ))
    end

    n_conexiones = n_capas - 1

    # Verificar que hay una matriz de pesos por cada conexión entre capas
    if length(pesos) != n_conexiones
        throw(ErrorAdaptador(
            "Se esperaban $n_conexiones matrices de pesos para $n_capas capas, " *
            "pero se encontraron $(length(pesos))",
            0
        ))
    end

    # Verificar que hay un vector de biases por cada conexión entre capas
    if length(biases) != n_conexiones
        throw(ErrorAdaptador(
            "Se esperaban $n_conexiones vectores de biases para $n_capas capas, " *
            "pero se encontraron $(length(biases))",
            0
        ))
    end

    # Verificar que hay una activación por cada capa de salida (conexión)
    if length(activaciones) != n_conexiones
        throw(ErrorAdaptador(
            "Se esperaban $n_conexiones funciones de activación para $n_capas capas, " *
            "pero se encontraron $(length(activaciones))",
            0
        ))
    end

    # Verificar dimensiones de cada capa
    for i in 1:n_conexiones
        n_in = capas[i]
        n_out = capas[i + 1]

        # Pesos: (n_out, n_in)
        peso_dims = size(pesos[i])
        if peso_dims != (n_out, n_in)
            throw(ErrorAdaptador(
                "Dimensiones de pesos en capa $i incompatibles: " *
                "se esperaba ($n_out, $n_in) pero se encontró $peso_dims",
                i
            ))
        end

        # Biases: (n_out,)
        bias_len = length(biases[i])
        if bias_len != n_out
            throw(ErrorAdaptador(
                "Dimensión de biases en capa $i incompatible: " *
                "se esperaba $n_out pero se encontró $bias_len",
                i
            ))
        end
    end

    nothing
end

"""
    adaptar_red(red::RedNeuronal, ::Type{T}) where {T<:AbstractFloat} -> RedBase{T}

Convierte una `RedNeuronal` (RandomCloud.jl) a `RedBase{T}` (DivisionNeuronal.jl).
Preserva pesos, biases y funciones de activación de cada capa.
Convierte el tipo numérico a `T` si es necesario.

# Argumentos
- `red`: red neuronal proveniente de RandomCloud.jl
- `T`: tipo numérico destino (`Float64`, `Float32`, etc.)

# Retorna
- `RedBase{T}` con la misma estructura y pesos convertidos a tipo `T`

# Errores
- `ErrorAdaptador` si la arquitectura es incompatible con la Fase_Division
"""
function adaptar_red(red::RedNeuronal, ::Type{T})::RedBase{T} where {T<:AbstractFloat}
    # Validar compatibilidad de arquitectura
    validar_arquitectura(red)

    # Extraer y convertir pesos a tipo T
    pesos_T = [T.(p) for p in red.pesos]

    # Extraer y convertir biases a tipo T
    biases_T = [T.(b) for b in red.biases]

    # Copiar capas y activaciones (no requieren conversión de tipo)
    capas = copy(red.capas)
    activaciones = copy(red.activaciones)

    # Construir RedBase{T}
    RedBase{T}(capas, pesos_T, biases_T, activaciones)
end

"""
    contar_parametros(red) -> Int

Cuenta el número total de parámetros (pesos + biases) de una red neuronal.
Funciona con cualquier red que tenga campos `pesos` y `biases` (duck-typing).

# Argumentos
- `red`: red neuronal con campos `pesos` (vector de matrices) y `biases` (vector de vectores)

# Retorna
- Número total de parámetros como `Int`
"""
function contar_parametros(red)::Int
    total = 0
    for p in red.pesos
        total += length(p)
    end
    for b in red.biases
        total += length(b)
    end
    total
end
