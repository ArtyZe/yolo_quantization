#include "upsample_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

layer make_upsample_layer(int batch, int w, int h, int c, int stride, int layer_quant_flag, int quant_stop_flag)
{
    layer l = {0};
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    if(stride < 0){
        stride = -stride;
        l.reverse=1;
        l.out_w = w/stride;
        l.out_h = h/stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_upsample_layer;
#ifdef QUANTIZATION
	l.activ_data_uint8_scales = calloc(1, sizeof(float));
    l.activ_data_uint8_zero_point = calloc(1, sizeof(uint8_t));
    l.min_activ_value = calloc(1, sizeof(float));
    l.max_activ_value = calloc(1, sizeof(float));
    l.output_uint8_final = calloc(l.batch*l.outputs, sizeof(uint8_t));

    l.layer_quant_flag = layer_quant_flag;
    l.quant_stop_flag = quant_stop_flag;
    if(l.layer_quant_flag && !l.close_quantization){
        l.forward = forward_upsample_layer_quant;
    }
    else{
        l.forward = forward_upsample_layer;
    }
#endif
    l.backward = backward_upsample_layer;
    #ifdef GPU
    l.forward_gpu = forward_upsample_layer_gpu;
    l.backward_gpu = backward_upsample_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    if(l.reverse) fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_upsample_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w*l->stride;
    l->out_h = h*l->stride;
    if(l->reverse){
        l->out_w = w/l->stride;
        l->out_h = h/l->stride;
    }
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->h*l->w*l->c;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}

void forward_upsample_layer(const layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if(l.reverse){
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
    }else{
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
    }
}

void forward_upsample_layer_quant(const layer l, network net)
{
    fill_cpu_uint8(l.outputs*l.batch, 0, l.output_uint8_final, 1);
    if(l.reverse){
        upsample_quant_cpu(l.output_uint8_final, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_uint8);
    }else{
        upsample_quant_cpu(net.input_uint8, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_uint8_final);
    }
    if(l.quant_stop_flag){
        printf("dequant from uint8 to float32 in layer %d\n", l.count);
        for (int s = 0; s < l.out_c; ++s) {
            for (int t = 0; t < l.out_w*l.out_h; ++t){
                int out_index = s*l.out_w*l.out_h + t;
                l.output[out_index] = (l.output_uint8_final[out_index] -  l.activ_data_uint8_zero_point[0]) * l.activ_data_uint8_scales[0];
            }
        }
    }
}

void backward_upsample_layer(const layer l, network net)
{
    if(l.reverse){
        upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta);
    }else{
        upsample_cpu(net.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta);
    }
}

#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.reverse){
        upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_gpu);
    }else{
        upsample_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
    }
    if(net.train && l.layer_quant_flag){
        assert(l.reverse == 0);
        cuda_pull_array(l.output_gpu, l.output, l.out_c*l.out_w*l.out_h);
        uint8_t input_fake_quant = 0;
        fake_quant_with_min_max_channel(1, l.output, &input_fake_quant, l.out_c*l.out_w*l.out_h, l.min_activ_value, l.max_activ_value, 
                                        l.activ_data_uint8_scales, l.activ_data_uint8_zero_point, ACTIV_QUANT, 0.999);
        assert(l.activ_data_uint8_scales[0] > 0);
        cuda_push_array(l.output_gpu, l.output, l.out_c*l.out_w*l.out_h);
    }
}

void backward_upsample_layer_gpu(const layer l, network net)
{
    if(l.reverse){
        upsample_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta_gpu);
    }else{
        upsample_gpu(net.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta_gpu);
    }
}
#endif
