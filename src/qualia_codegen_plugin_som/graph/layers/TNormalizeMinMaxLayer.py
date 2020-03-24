from __future__ import annotations

import sys
from dataclasses import dataclass

from qualia_codegen_core.graph.layers import TBaseLayer
from qualia_codegen_core.typing import TYPE_CHECKING, NDArrayFloatOrInt

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: TCH003

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@dataclass
class TNormalizeMinMaxLayer(TBaseLayer):
    minimum: NDArrayFloatOrInt
    reciprocal_divisor: NDArrayFloatOrInt

    @property
    @override
    def weights(self) -> OrderedDict[str, NDArrayFloatOrInt]:
        w = super().weights
        w['minimum'] = self.minimum
        w['reciprocal_divisor'] = self.reciprocal_divisor
        return w
