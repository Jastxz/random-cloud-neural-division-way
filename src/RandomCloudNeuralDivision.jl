module RandomCloudNeuralDivision

using RandomCloud
using DivisionNeuronal
using JSON3
using StructTypes
using KernelAbstractions
using Random

# Tipos base, configuración y errores
include("tipos.jl")

# Validación de datos de entrada
include("validacion.jl")

# Conversión RedNeuronal → RedBase
include("adaptador.jl")

# Selección CPU/GPU y transferencias
include("backend.jl")

# Pre-alocación y gestión de buffers
include("buffers.jl")

# Motor del pipeline: ejecutar_pipeline, fases individuales
include("motor.jl")

# resumen() y lógica de Informe_RCND
include("informe.jl")

# Serialización/deserialización JSON
include("serializacion.jl")

# API pública
export Configuracion_RCND,
       Motor_RCND,
       Informe_RCND,
       TopologiaOptima,
       ejecutar_pipeline,
       ejecutar_fase_nube,
       ejecutar_fase_division,
       serializar,
       deserializar,
       resumen

end # module RandomCloudNeuralDivision
