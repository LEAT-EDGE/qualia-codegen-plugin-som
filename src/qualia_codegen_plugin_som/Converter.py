# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path
from typing import ClassVar

import qualia_codegen_core
from qualia_codegen_core.typing import TYPE_CHECKING

from .graph import layers

if TYPE_CHECKING:
    from qualia_codegen_core.graph.layers import TBaseLayer  # noqa: TCH002

logger = logging.getLogger(__name__)

class Converter(qualia_codegen_core.Converter):
    layer_template_files: ClassVar[dict[type[TBaseLayer], str | None]] = {**qualia_codegen_core.Converter.layer_template_files,
        # SOM layers
        layers.TNormalizeMinMaxLayer: 'normalizeminmax',
        layers.TDSOMLayer: 'dsom',
        layers.TSOMLabellingLayer: 'somlabelling',
    }

    TEMPLATE_PATH = files('qualia_codegen_plugin_som.assets')

    def __init__(self, output_path: Path | None = None) -> None:
        super().__init__(output_path=output_path)

        # Super failed to popuate template_path
        if self._template_path is None:
            return

        template_path: Path | None = None
        # Prepend to template_path so that our files have higher priority over qualia_codegen_core files
        if isinstance(Converter.TEMPLATE_PATH, Path): # Already Path objected, no need for hackery
            template_path = Converter.TEMPLATE_PATH
        elif sys.version_info >= (3, 10): # Python 3.10 may return MultiplexedPath
            from importlib.readers import MultiplexedPath
            if isinstance(Converter.TEMPLATE_PATH, MultiplexedPath):
                template_path = Converter.TEMPLATE_PATH / '' # / operator applies to underlying Path

        if template_path is not None:
            self._template_path.insert(0, template_path)
        else: # If we failed, also clear _template_path to fail conversion altogether instead of having incorrect search path
            self._template_path = None
