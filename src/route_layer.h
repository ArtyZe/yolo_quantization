#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"
#include "layer.h"
#include "assert.h"

typedef layer route_layer;

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes, int layer_quant_flag, int quant_stop_flag, int close_quantization);
void forward_route_layer(const route_layer l, network net);
void forward_route_layer_quant(const route_layer l, network net);
void backward_route_layer(const route_layer l, network net);
void resize_route_layer(route_layer *l, network *net);

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net);
void backward_route_layer_gpu(const route_layer l, network net);
#endif

#endif
