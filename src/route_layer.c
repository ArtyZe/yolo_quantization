#include "route_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes, int layer_quant_flag, int quant_stop_flag)
{
    fprintf(stderr,"route ");
    route_layer l = {0};
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    l.delta =  calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));

    l.layer_quant_flag = layer_quant_flag;
    l.quant_stop_flag = quant_stop_flag;

    if(l.layer_quant_flag){
        l.forward = forward_route_layer_quant;
    }else{
        l.forward = forward_route_layer;
    }
    l.backward = backward_route_layer;
    #ifdef GPU
    l.forward_gpu = forward_route_layer_gpu;
    l.backward_gpu = backward_route_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    return l;
}

void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}

void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}

void forward_route_layer_quant(const route_layer l, network net)
{
    int i, j, k;
    int offset = 0;
    uint16_t *temp_output_uint16 = malloc(l.outputs * sizeof(uint16_t));
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        uint8_t *input_uint8 = net.layers[index].output_uint8_final;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_cpu_uint8(input_size, input_uint8 + j*input_size, 1, temp_output_uint16 + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
    for(k = 0; k < l.outputs; ++k){
        assert(temp_output_uint16[k] > QUANT_NEGATIVE_LIMIT && temp_output_uint16[k] < QUANT_POSITIVE_LIMIT);
    }
    if(l.quant_stop_flag){
        for (int s = 0; s < l.out_c; ++s) {
            for (int t = 0; t < l.out_w*l.out_h; ++t){
                int out_index = s*l.out_w*l.out_h + t;
                l.output[out_index] = (l.output_uint8_final[out_index] -  l.activ_data_int8_zero_point[0]) * l.activ_data_uint8_scales[0];
            }
        }
    }
}

void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_gpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
    if(net.train && l.layer_quant_flag){
        cuda_pull_array(l.output_gpu, l.output, l.out_c*l.out_w*l.out_h);
        uint8_t input_fake_quant = 0;
        fake_quant_with_min_max_channel(1, l.output, &input_fake_quant, l.out_c*l.out_w*l.out_h, l.min_activ_value, l.max_activ_value, 
                                        l.activ_data_uint8_scales, l.activ_data_int8_zero_point, ACTIV_QUANT, 0.999);
        assert(l.activ_data_uint8_scales[0] > 0);
        cuda_push_array(l.output_gpu, l.output, l.out_c*l.out_w*l.out_h);
        cuda_push_array(l.activ_data_int8_scales_gpu, l.activ_data_uint8_scales, 1);
        cuda_push_array_int8(l.activ_data_int8_zero_point_gpu, l.activ_data_int8_zero_point, 1);
    }
}

void backward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#endif
