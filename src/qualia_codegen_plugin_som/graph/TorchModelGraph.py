from __future__ import annotations

import logging
import sys
from typing import Any, Callable, ClassVar, cast

import numpy as np
import qualia_codegen_core.graph
from qualia_codegen_core.graph.layers import TBaseLayer, TFlattenLayer
from qualia_core.typing import TYPE_CHECKING
from qualia_plugin_som.learningmodel.pytorch.layers import NormalizeMinMax, SOMLabelling
from qualia_plugin_som.learningmodel.pytorch.layers.som import DSOM, KSOM

from .layers import (
    TDSOMLayer,
    TNormalizeMinMaxLayer,
    TSOMLabellingLayer,
)

if TYPE_CHECKING:
    from qualia_codegen_core.graph import ModelGraph  # noqa: TCH002
    from qualia_codegen_core.graph.LayerNode import LayerNode  # noqa: TCH002
    from torch.nn import Module  # noqa: TCH002

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class TorchModelGraph(qualia_codegen_core.graph.TorchModelGraph):
    MODULE_MAPPING: ClassVar[dict[type[Module], Callable[[Module, TBaseLayer], tuple[type[TBaseLayer], list[Any]]]]] = {
        # SOM layers
        NormalizeMinMax: lambda module, _: (TNormalizeMinMaxLayer,
                                         [cast(NormalizeMinMax, module).min.detach().numpy(),
                                          cast(NormalizeMinMax, module).reciprocal_divisor.detach().numpy()]),
        # DSOM might require reformatting weights for channels_first/last, 2D grid only
        DSOM: lambda module, _: (TDSOMLayer, [cast(DSOM, module).neurons.detach().numpy(),
                                           cast(DSOM, module).out_features[0],
                                           cast(DSOM, module).out_features[1],
                                           cast(DSOM, module).learning_rate.detach().numpy(),
                                           cast(DSOM, module).elasticity_squared.detach().numpy()]),
        # KSOM might require reformatting weights for channels_first/last, 2D grid only
        KSOM: lambda module, _: (TDSOMLayer, [cast(KSOM, module).neurons.detach().numpy(),
                                           cast(KSOM, module).out_features[0],
                                           cast(KSOM, module).out_features[1],
                                           np.array(0),
                                           np.array(0)]),
        # SOMLabelling labels are integers, convert to uint8
        SOMLabelling: lambda module, _: (TSOMLabellingLayer, [cast(SOMLabelling, module).labels.byte().detach().numpy()]),
    }

    @override
    def convert(self,
                custom_layers: dict[type[Module],
                                             Callable[[Module, TBaseLayer],
                                                      tuple[type[TBaseLayer], list[Any]]]] | None = None) -> ModelGraph | None:
        custom_layers = custom_layers if custom_layers is not None else {}
        custom_layers = {**TorchModelGraph.MODULE_MAPPING, **custom_layers}

        ret = super().convert(custom_layers)

        # channels_first to channels_last for first SOM layer
        if not self.__reformat_som_weights_data():
            logger.error('Conversion failed')
            return None

        return ret

    def __reformat_som(self, somnode: LayerNode, flattennode: LayerNode) -> bool:
        if not isinstance(somnode.layer, TDSOMLayer):
            return False
        som = somnode.layer
        # reshape using Flatten input shape (for example last Conv output)
        som.neurons = som.neurons.reshape(
                (som.neurons.shape[0], ) + flattennode.layer.input_shape[0][-1:] + flattennode.layer.input_shape[0][1:-1])
        som.neurons = TorchModelGraph.transpose(som.neurons)
        som.neurons = som.neurons.reshape((som.neurons.shape[0], -1))
        return True

    def __reformat_som_weights_data(self) -> bool:
        """After Flatten comes SOM, must reshape weights and swap axes for channels_last used by C code."""
        somnodes = [n for n in self.nodes if isinstance(n.layer, TDSOMLayer)]
        for somnode in somnodes:
            if len(somnode.innodes) != 1:
                logger.error('SOM %s should only have one input layer, got: %s', somnode.layer.name, len(somnode.innodes))
                return False

            innode = somnode.innodes[0]
            # Before SOM may come NormalizeMinMax, reformat SOM according to previous layer which should be Flatten
            if isinstance(innode.layer, TNormalizeMinMaxLayer):
                if len(innode.innodes) != 1:
                    logger.error('NormalizeMinMax %s should only have one input layer, got: %s',
                                 innode.layer.name, len(innode.innodes))
                    return False
                innode = innode.innodes[0]

            # No flatten before the SOM layer, may not need reformatting
            if not isinstance(innode.layer, TFlattenLayer):
                logger.info('No flatten before SOM layer %s, not reformatting SOM', somnode.layer.name)
                continue

            if not self.__reformat_som(somnode, innode):
                logger.error('Reformatting SOM %s failed', somnode.layer.name)
                return False

            logger.info('Reformatted SOM %s', somnode.layer.name)
        return True
