/**
  ******************************************************************************
  * @file    weights/normalizeminmax.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 january 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const {{ weights.minimum.dtype }} {{ node.layer.name }}_minimum = {{ weights.minimum.data }};
const {{ weights.reciprocal_divisor.dtype }} {{ node.layer.name }}_reciprocal_divisor = {{ weights.reciprocal_divisor.data }};
