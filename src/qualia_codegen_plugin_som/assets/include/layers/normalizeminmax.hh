/**
  ******************************************************************************
  * @file    normalizeminmax.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES {{ node.input_shape[0][-1] }}

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[INPUT_SAMPLES];

#if 0
void {{ node.layer.name }}(
  const number_t input[INPUT_SAMPLES], 			      // IN
  const number_t min_val,
  const number_t reciprocal_divisor,
	number_t output[INPUT_SAMPLES]);			                // OUT
#endif

#undef INPUT_SAMPLES

#endif//_{{ node.layer.name | upper }}_H_
