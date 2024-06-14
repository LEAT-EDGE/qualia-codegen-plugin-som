/**
  ******************************************************************************
  * @file    weights/dsom.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    18 january 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES {{ node.input_shape[0][-1] }}
#define NEURONS {{ node.layer.neurons.shape[0] }}

//#define ONLINE_LEARNING

#ifndef ONLINE_LEARNING
const
#endif
  {{ weights.neurons.dtype }} {{ node.layer.name }}_neurons[NEURONS][INPUT_SAMPLES] = {{ weights.neurons.data }};
const {{ weights.learning_rate.dtype }} {{ node.layer.name }}_learning_rate = {{ weights.learning_rate.data }};
const {{ weights.elasticity_squared.dtype }} {{ node.layer.name }}_elasticity_squared = {{ weights.elasticity_squared.data }};

#undef INPUT_SAMPLES
#undef NEURONS
