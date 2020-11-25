#ifndef UPSAMPLE_LAYER_H
#define UPSAMPLE_LAYER_H
#include "darknet.h"
#include "assert.h"

layer make_upsample_layer(int batch, int w, int h, int c, int stride, int layer_quant_flag, int quant_stop_flag);
void forward_upsample_layer(const layer l, network net);
void forward_upsample_layer_quant(const layer l, network net);
void backward_upsample_layer(const layer l, network net);
void resize_upsample_layer(layer *l, int w, int h);

#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net);
void backward_upsample_layer_gpu(const layer l, network net);
#endif

#endif
