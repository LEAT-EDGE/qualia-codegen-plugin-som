from qualia_codegen_core.graph.layers import layers_t as core_layers_t

# SOM layers
from .TDSOMLayer import TDSOMLayer
from .TNormalizeMinMaxLayer import TNormalizeMinMaxLayer
from .TSOMLabellingLayer import TSOMLabellingLayer

layers_t = {**core_layers_t,
    # SOM layers
    'TDSOMLayer': TDSOMLayer,
    'TNormalizeMinMaxLayer': TNormalizeMinMaxLayer,
    'TSOMLabellingLayer': TSOMLabellingLayer,
}

__all__ = ['TDSOMLayer', 'TNormalizeMinMaxLayer', 'TSOMLabellingLayer']
