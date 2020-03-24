/**
  ******************************************************************************
  * @file    weights/somlabelling.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    18 january 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

// 2D grid
#define GRID_WIDTH {{ node.layer.labels.shape[0] }} // BMU x
#define GRID_HEIGHT {{ node.layer.labels.shape[1] }} // BMU y
#define CLASSES {{ node.layer.labels.shape[-1] }} // label

const {{ weights.labels.dtype }} {{ node.layer.name }}_labels[GRID_WIDTH][GRID_HEIGHT][CLASSES] = {{ weights.labels.data }};

#undef NEURONS
