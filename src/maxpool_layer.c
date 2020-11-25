#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding, int layer_quant_flag, int quant_stop_flag)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));

    l.backward = backward_maxpool_layer;
    l.forward = forward_maxpool_layer;
#ifdef QUANTIZATION
	l.activ_data_uint8_scales = calloc(1, sizeof(float));
    l.activ_data_uint8_zero_point = calloc(1, sizeof(uint8_t));
    l.min_activ_value = calloc(1, sizeof(float));
    l.max_activ_value = calloc(1, sizeof(float));
    l.output_uint8_final = calloc(l.batch*l.outputs, sizeof(uint8_t));
    l.layer_quant_flag = layer_quant_flag;
    l.quant_stop_flag = quant_stop_flag;
    l.input_uint8 = calloc(output_size, sizeof(uint8_t));
    if(l.layer_quant_flag && !l.close_quantization){
        l.forward = forward_maxpool_layer_quant;
    }
    else{
        l.forward = forward_maxpool_layer;
    }
#endif
    #ifdef GPU
        // #ifdef QUANTIZATION
        //     l.min_activ_value_gpu = cuda_make_array(l.min_activ_value, 1);
        //     l.min_activ_value_gpu = cuda_make_array(l.min_activ_value, 1);
        //     l.activ_data_uint8_scales_gpu = cuda_make_array(l.activ_data_uint8_scales, 1);
        //     l.activ_data_uint8_zero_point_gpu = cuda_make_array(l.activ_data_uint8_zero_point, 1);
        // #endif
        l.forward_gpu = forward_maxpool_layer_gpu;
        l.backward_gpu = backward_maxpool_layer_gpu;
        l.indexes_gpu = cuda_make_int_array(0, output_size);
        l.output_gpu  = cuda_make_array(l.output, output_size);
        l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

void forward_maxpool_layer_quant(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    #pragma omp parallel for
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            uint8_t val = (valid != 0) ? net.input_uint8[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output_uint8_final[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
    if(l.quant_stop_flag){
        printf("dequant from uint8 to float32 in layer %d\n", l.count);
        #pragma omp parallel for
        for (int s = 0; s < l.out_c; ++s) {
            for (int t = 0; t < l.out_w*l.out_h; ++t){
                int out_index = s*l.out_w*l.out_h + t;
                l.output[out_index] = (l.output_uint8_final[out_index] -  l.activ_data_uint8_zero_point[0]) * l.activ_data_uint8_scales[0];
            }
        }
    }
}

void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;    
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

