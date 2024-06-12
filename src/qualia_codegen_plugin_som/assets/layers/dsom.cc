/**
  ******************************************************************************
  * @file    dsom.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    18 january 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif
//TODO: get rid of math.h for pow2_roots generation in fixed-point
#include <math.h>
#include <stdlib.h>

#define INPUT_SAMPLES {{ node.input_shape[0][-1] }}
#define NEURONS       {{ node.layer.neurons.shape[0] }}
#define GRID_WIDTH    {{ node.layer.width }}
#define GRID_HEIGHT   {{ node.layer.height }}

//#define ONLINE_LEARNING

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR {{ node.q.weights_scale_factor }}
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define TMP_SCALE_FACTOR {{ [node.q.weights_scale_factor, node.innodes[0].q.output_scale_factor] | max }}
#define OUTPUT_ROUND_MODE ROUND_MODE_{{ node.q.output_round_mode | upper }}
// Scale factor to use for the exp computation, equals TMP_SCALE_FACTOR here
{% set exp_scale_factor = [node.q.weights_scale_factor, node.innodes[0].q.output_scale_factor] | max -%}
#define EXP_SCALE_FACTOR {{ exp_scale_factor }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}

/*
 * e^x in fixed-point, method from https://medium.com/coinmonks/math-in-solidity-part-5-exponent-and-logarithm-9aef8515136e
 */
{% if node.q.number_type.__name__ == 'int' %}
// Lookup-table for fixed-point sqrt^i(2)
static const LONG_NUMBER_T pow2_roots[EXP_SCALE_FACTOR] = {
  // Generate for each bit of EXP_SCALE_FACTOR
{%- for i in range(exp_scale_factor) %}
  (LONG_NUMBER_T)floor({% for j in range(i + 1) -%} sqrt( {%- endfor -%} 2 {%- for j in range(i + 1) -%} ) {%- endfor %} * (1 << EXP_SCALE_FACTOR)),
{%- endfor %}
};

static inline LONG_NUMBER_T pow2_fixed(LONG_NUMBER_T x) {
  LONG_NUMBER_T x_int_part = x >> EXP_SCALE_FACTOR;

  if (x_int_part < -EXP_SCALE_FACTOR) { // underflow
    return 0;
  }

  // output scale factor = 15 bits (EXP_SCALE_FACTOR)
  // roots scale factor = 15 bits (EXP_SCALE_FACTOR)
  // product of roots = 1 bit (prod i=0→+inf sqrt^i(2) = 2)
  if (x_int_part >= (LONG_NUMBER_T)(sizeof(LONG_NUMBER_T) * 8 - (EXP_SCALE_FACTOR * 2 + 1))) { // overflow
    return ((LONG_NUMBER_T)1) << (sizeof(LONG_NUMBER_T) * 8 - (EXP_SCALE_FACTOR + 1)); // max value with EXP_SCALE_FACTO scale factor
  }

  // Compute integer part of x: 2^⌊x⌋ with EXP_SCALE_FACTOR scale factor
  LONG_NUMBER_T y = 1 << EXP_SCALE_FACTOR;
  if (x_int_part < 0) {
    y = (y >> -x_int_part);
  } else {
    y = (y << x_int_part);
  }

  // Compute fractional part of x
  {%- for i in range(exp_scale_factor) %}
  if (x & 0b1{% for j in range(exp_scale_factor - i - 1) %}0{% endfor %}) {
    y = (y * pow2_roots[{{ i }}]) >> EXP_SCALE_FACTOR;
  }
  {%- endfor %}

  return y;
}

static inline LONG_NUMBER_T exp_fixed(LONG_NUMBER_T x) {
  // e^x = 2^(x * log2(e))
  static const LONG_NUMBER_T log2e_fixed = (LONG_NUMBER_T)floor(log2(exp(1)) * (1 << EXP_SCALE_FACTOR));
  return pow2_fixed(((LONG_NUMBER_T)(x) * log2e_fixed) >> EXP_SCALE_FACTOR);
}
{% endif %}
/*
 *
 */



static inline LONG_NUMBER_T euclidean_distance_squared(
  const NUMBER_T input[INPUT_SAMPLES],
  const NUMBER_T neuron[INPUT_SAMPLES]) {

  static unsigned int i;
  static LONG_NUMBER_T diff;
  static LONG_NUMBER_T dist;

  dist = 0;
  
  // Compute squared distance to input
  for (i = 0; i < INPUT_SAMPLES; i++) {
    diff = scale(NUMBER_T, input[i], INPUT_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE) - scale(NUMBER_T, neuron[i], WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    //Warning: overflow in fixed-point? Scale back to somewhat mitigate
    dist += scale(NUMBER_T, diff * diff, TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
  }
  return dist;
}

static inline unsigned short manhattan_distance(const unsigned short a, const unsigned short b) {
  static unsigned short ax, ay, bx, by;

  ax = a % GRID_WIDTH;
  ay = a / GRID_WIDTH;
  bx = b % GRID_WIDTH;
  by = b / GRID_WIDTH;
  return abs(bx - ax) + abs(by - ay);
}

static inline void {{ node.layer.name }}(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
#ifndef ONLINE_LEARNING
  const 
#endif
    NUMBER_T neurons[NEURONS][INPUT_SAMPLES],                // IN
  const NUMBER_T learning_rate,                    // IN
  const NUMBER_T elasticity_squared,              // IN
	unsigned short output[2]) {			          // OUT

  static LONG_NUMBER_T distances_to_input_squared[NEURONS];
  unsigned short k;
  unsigned short bmu = 0;
#ifdef ONLINE_LEARNING
  unsigned short i;
  unsigned short distance_to_bmu;
  LONG_NUMBER_T learning;
  LONG_NUMBER_T neighbourhood;
#endif

  distances_to_input_squared[0] = euclidean_distance_squared(input, neurons[0]);

  // Find BMU
  for (k = 1; k < NEURONS; k++) {
    distances_to_input_squared[k] = euclidean_distance_squared(input, neurons[k]);

    // BMU has min distance to input
    if (distances_to_input_squared[k] < distances_to_input_squared[bmu]) {
      bmu = k;
    }
  }

#ifdef ONLINE_LEARNING
  // Online learning
  if (distances_to_input_squared[bmu] != 0) {
    for (k = 0; k < NEURONS; k++) {
      distance_to_bmu = manhattan_distance(k, bmu);

{% if node.q.number_type.__name__ == 'int' %}
      LONG_NUMBER_T numerator = -scale(NUMBER_T, (LONG_NUMBER_T)(distance_to_bmu) * distance_to_bmu, -(2 * TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE); // Scale up before division
      LONG_NUMBER_T denominator = scale(NUMBER_T, (LONG_NUMBER_T)(elasticity_squared) * distances_to_input_squared[bmu], WEIGHTS_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      neighbourhood = exp_fixed(numerator / denominator);
      learning = scale(NUMBER_T, distances_to_input_squared[k] * neighbourhood, EXP_SCALE_FACTOR, OUTPUT_ROUND_MODE); // Scale neighbourhood back
      learning = scale(NUMBER_T, learning * learning_rate, TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE); // Scale distances_to_input_squared back, keep WEIGHTS_SCALE_FACTOR format
{% else %}
      neighbourhood = exp(-(distance_to_bmu * distance_to_bmu) / (elasticity_squared * distances_to_input_squared[bmu]));
      learning = learning_rate * distances_to_input_squared[k] * neighbourhood;
{% endif %}

      for (i = 0; i < INPUT_SAMPLES; i++) {
        neurons[k][i] += clamp_to(NUMBER_T,
                            scale(NUMBER_T,
                              learning * (scale(NUMBER_T, input[i], INPUT_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE) - scale(NUMBER_T, neurons[k][i], WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE)),
                              TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE));
      }
    }
  }

#endif

  // Convert BMU number to position in 2D grid
  // Reversed in unravel_index PyTorch
  output[1] = bmu % GRID_WIDTH; // x
  output[0] = bmu / GRID_WIDTH; // y
}

#undef INPUT_SAMPLES
#undef NEURONS
#undef GRID_WIDTH
#undef GRID_HEIGHT
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_ROUND_MODE
#undef TMP_SCALE_FACTOR
#undef EXP_SCALE_FACTOR
