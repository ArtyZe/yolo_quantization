#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabsf(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void forward_convolutional_layer_quant_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    copy_gpu(l.nweights*l.batch, l.weights_gpu, 1, l.weights_bn_backup_gpu, 1);
    copy_gpu(l.out_c*l.batch, l.biases_gpu, 1, l.biases_bn_backup_gpu, 1);
#ifdef QUANTIZATION

//////////////////////////////////////////////////////////////////
//    this process in training graph is to get
//    the output mean and variance then use to 
//    fold the batch norm parameters
//////////////////////////////////////////////////////////////////
    if(net.train && l.batch_normalize && l.layer_quant_flag){
        fill_gpu(l.outputs*l.batch, 0, l.output_bn_backup_gpu, 1);

        int i1, j1;
        int m1 = l.n/l.groups;
        int k1 = l.size*l.size*l.c/l.groups;
        int n1 = l.out_w*l.out_h;
        for(i1 = 0; i1 < l.batch; ++i1){
            for(j1 = 0; j1 < l.groups; ++j1){
                float *a1 = l.weights_gpu + j1*l.nweights/l.groups;
                float *b1 = net.workspace;
                float *c1 = l.output_bn_backup_gpu + (i1*l.groups + j1)*n1*m1;
                float *im1 = net.input_gpu + (i1*l.groups + j1)*l.c/l.groups*l.h*l.w;

                if (l.size == 1){
                    b1 = im1;
                } else {
                    im2col_gpu(im1, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b1);
                }
                gemm_gpu(0,0,m1,n1,k1,1,a1,k1,b1,n1,1,c1,n1);
            }
        }
    }
    // flod batchnorm with conv
    if(l.batch_normalize && l.layer_quant_flag){ 
        forward_batchnorm_layer_quant_gpu(l, net);
    }
    if(net.train && l.layer_quant_flag){
        float min_weights_value = 0;
        float max_weights_value = 1;
        // calculate input quantization scale and zeropoint s1, z1
        cuda_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
        cuda_pull_array(net.input_gpu, net.input, l.c*l.w*l.h);
        if(l.count == 0){
            fake_quant_with_min_max_channel(1, net.input, l.weights_uint8, l.c*l.w*l.h, l.min_input_value, l.max_input_value, 
                l.input_data_uint8_scales, l.input_data_uint8_zero_point, INPUT_QUANT, 0.9);
            assert(l.input_data_uint8_scales[0] > 0);
        }

        // calculate weights quantization scale and zeropoint s2, z2
        fake_quant_with_min_max_channel(1, l.weights, l.weights_uint8, l.n*l.c*l.size*l.size, &min_weights_value, &max_weights_value, 
                                   l.weight_data_uint8_scales, l.weight_data_uint8_zero_point, WEIGHT_QUANT, 0.9);
        assert(l.weight_data_uint8_scales[0] > 0);
        cuda_push_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
        // cuda_push_array(l.weight_data_uint8_scales_gpu, l.weight_data_uint8_scales, 1);
        // cuda_push_array_int8(l.weight_data_uint8_zero_point_gpu, l.weight_data_uint8_zero_point, 1);
        cuda_push_array_int8(l.weights_quant_gpu, l.weights_uint8, l.c*l.n*l.size*l.size);
    }   
#endif
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
#ifndef QUANTIZATION    
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    }else{
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
#else
    if (l.batch_normalize && !l.layer_quant_flag) {
        forward_batchnorm_layer_gpu(l, net);
    }else{
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
#endif    

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    // activation quantization
    if(net.train && l.layer_quant_flag){
        cuda_pull_array(l.output_gpu, l.output, l.out_c*l.out_w*l.out_h);
        uint8_t input_fake_quant = 0;
        fake_quant_with_min_max_channel(1, l.output, &input_fake_quant, l.out_c*l.out_w*l.out_h, l.min_activ_value, l.max_activ_value, 
                                        l.activ_data_uint8_scales, l.activ_data_uint8_zero_point, ACTIV_QUANT, 0.999);
        assert(l.activ_data_uint8_scales[0] > 0);
        cuda_push_array(l.output_gpu, l.output, l.out_c*l.out_w*l.out_h);
        // cuda_push_array(l.activ_data_uint8_scales_gpu, l.activ_data_uint8_scales, 1);
        // cuda_push_array_int8(l.activ_data_uint8_zero_point_gpu, l.activ_data_uint8_zero_point, 1);	
    }
    
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
            
#ifndef QUANTIZATION    
    if (l.batch_normalize) {
        backward_batchnorm_layer_gpu(l, net);
    }else{
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
#else
    if(l.batch_normalize && !l.layer_quant_flag){
        backward_batchnorm_layer_gpu(l, net);
    }else{
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
#endif 

    // float *original_input = net.input_gpu;

    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if(net.delta_gpu) {
                if (l.binary || l.xnor) swap_binary(&l);
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
            }
        }
    }
   
#ifdef QUANTIZATION  
    if(l.batch_normalize && l.layer_quant_flag){
        backward_batchnorm_layer_quant_gpu(l, net);
    }
#endif
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
// #ifdef QUANTIZATION
//     cuda_pull_array(l.activ_data_uint8_scales_gpu, l.activ_data_uint8_scales, 1);
//     cuda_pull_array_int8(l.activ_data_uint8_zero_point_gpu, l.activ_data_uint8_zero_point, 1);	
//     cuda_pull_array(l.weight_data_uint8_scales_gpu, l.weight_data_uint8_scales, 1);
//     cuda_pull_array_int8(l.weight_data_uint8_zero_point_gpu, l.weight_data_uint8_zero_point, 1);	
//     cuda_pull_array_int8(l.weights_quant_gpu, l.weights_uint8, l.c*l.n*l.size*l.size);
// #endif
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
// #ifdef QUANTIZATION
//     cuda_push_array(l.activ_data_uint8_scales_gpu, l.activ_data_uint8_scales, 1);
//     cuda_push_array_int8(l.activ_data_uint8_zero_point_gpu, l.activ_data_uint8_zero_point, 1);	
//     cuda_push_array(l.weight_data_uint8_scales_gpu, l.weight_data_uint8_scales, 1);
//     cuda_push_array_int8(l.weight_data_uint8_zero_point_gpu, l.weight_data_uint8_zero_point, 1);	
//     cuda_push_array_int8(l.weights_quant_gpu, l.weights_uint8, l.c*l.n*l.size*l.size);
// #endif
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

int cmp( const void *a , const void *b )
{
		return *(float *)a > *(float *)b ? 1 : -1; 
}
void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
#ifdef PRUNE
    int size = l.size*l.size*l.c/l.groups*l.n;
    prune_gpu(size,l.weights_gpu,l.weight_updates_gpu,0.001,1);
#endif

#ifdef QUANTIZATION
    axpy_gpu(l.nweights, -decay*batch, l.weights_bn_backup_gpu, 1, l.weight_updates_gpu, 1);
    axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_bn_backup_gpu, 1);
    scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

    
    axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_bn_backup_gpu, 1);
    scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

    copy_gpu(l.nweights, l.weights_bn_backup_gpu, 1, l.weights_gpu, 1);
    copy_gpu(l.out_c, l.biases_bn_backup_gpu, 1, l.biases_gpu, 1);
#else
    axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

    
    axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);
#endif
    if(l.scales_gpu && !l.layer_quant_flag){			      	
        axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
        scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
    }
}


