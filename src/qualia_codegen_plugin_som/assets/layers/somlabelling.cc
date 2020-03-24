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

// 2D grid
#define GRID_WIDTH {{ node.layer.labels.shape[0] }} // BMU x
#define GRID_HEIGHT {{ node.layer.labels.shape[1] }} // BMU y
#define CLASSES {{ node.output_shape[0][-1] }} // label

// For fixed point quantization
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}


static inline void {{ node.layer.name }}(
  const unsigned short input[2], 			      // IN
  const uint8_t labels[GRID_WIDTH][GRID_HEIGHT][CLASSES],           // IN
	NUMBER_T output[CLASSES]) {			                // OUT

  unsigned int i;

  for (i = 0; i < CLASSES; i++) {
    if (labels[input[0]][input[1]][i]) {
      output[i] = 1;
    } else {
      output[i] = 0;
    }
  }
}

#undef GRID_WIDTH
#undef GRID_HEIGHT
#undef CLASSES
