/**
  ******************************************************************************
  * @file    normalizeminmax.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    18 january 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#define INPUT_SAMPLES {{ node.input_shape[0][-1] }}

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR {{ node.q.weights_scale_factor }}
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define TMP_SCALE_FACTOR {{ [node.q.weights_scale_factor, node.q.output_scale_factor] | max }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}


static inline void {{ node.layer.name }}(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
  const NUMBER_T min_val,
  const NUMBER_T reciprocal_divisor,
	NUMBER_T output[INPUT_SAMPLES]) {			                // OUT

  unsigned int i;
  LONG_NUMBER_T tmp;

  for (i = 0; i < INPUT_SAMPLES; i++) {
    tmp = scale(NUMBER_T, input[i], INPUT_SCALE_FACTOR - TMP_SCALE_FACTOR) - scale(NUMBER_T, min_val, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR);
    tmp = tmp * reciprocal_divisor;

    output[i] = clamp_to(NUMBER_T, scale(NUMBER_T, tmp, TMP_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR));
  }
}

#undef INPUT_SAMPLES
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
