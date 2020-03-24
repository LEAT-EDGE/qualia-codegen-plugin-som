/**
  ******************************************************************************
  * @file    somlabelling.hh
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

#define CLASSES {{ node.output_shape[0][-1] }} // label

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[CLASSES];

#if 0
void {{ node.layer.name }}(
  const unsigned short input[2], 			      // IN
  const uint8_t labels[GRID_WIDTH][GRID_HEIGHT][CLASSES],           // IN
	number_t output[CLASSES]);			                // OUT
#endif

#undef CLASSES

#endif//_{{ node.layer.name | upper }}_H_
