#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
// #include "cblas.h"
#include "mkl.h"
#include "mkl_cblas.h"
#ifdef OPENBLAS
    #include "mkl.h"
    #include "mkl_cblas.h"
#endif
#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, 
                                             int batch_normalize, int binary, int quant_stop_flag, int adam, int close_quantization, int layer_quantization, int count)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = 0;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));	

    float scale = sqrt(2./(size*size*c/l.groups));
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
#ifdef QUANTIZATION
    l.layer_quant_flag = layer_quantization;
    l.close_quantization = close_quantization;

    l.min_activ_value = calloc(1, sizeof(float));
    l.max_activ_value = calloc(1, sizeof(float));
    l.min_input_value = calloc(1, sizeof(float));
    l.max_input_value = calloc(1, sizeof(float));

    l.quant_stop_flag = quant_stop_flag;

	l.activ_data_uint8_scales = calloc(1, sizeof(float));
    l.weight_data_uint8_scales = calloc(l.n, sizeof(float));
    l.input_data_uint8_scales = calloc(1, sizeof(float));
	l.biases_data_uint8_scales = calloc(l.n, sizeof(float));

    l.input_sum_int = calloc(l.out_w*l.out_h*l.n, sizeof(uint32_t));
    l.weights_sum_int = calloc(l.n, sizeof(uint32_t));
    l.mult_zero_point = calloc(l.n, sizeof(uint32_t));

    l.M = calloc(l.n, sizeof(float));
    l.M0 = calloc(l.n, sizeof(int32_t));
    l.M_value = calloc(l.n, sizeof(double));
    l.M0_right_shift = calloc(l.n, sizeof(int));
    l.M0_right_shift_value = calloc(l.n, sizeof(double));

    l.activ_data_uint8_zero_point = calloc(1, sizeof(uint8_t));
    l.weight_data_uint8_zero_point = calloc(l.n, sizeof(uint8_t));
    l.input_data_uint8_zero_point = calloc(1, sizeof(uint8_t));
    l.biases_data_uint8_zero_point = calloc(l.n, sizeof(uint8_t));

    l.weights_uint8 = calloc(l.nweights, sizeof(uint8_t));
    l.weights_norm = calloc(l.nweights, sizeof(float));
	l.biases_int32 = calloc(l.n, sizeof(uint32_t));
    l.input_uint8 = calloc(l.c*l.w*l.h, sizeof(uint8_t));

    l.weights_int16 = calloc(l.nweights, sizeof(int16_t));
    l.input_int16 = calloc(l.inputs, sizeof(int16_t));
    l.zero_point_int16 = calloc(l.nweights, sizeof(int16_t));

    l.weights_bn_backup = calloc(l.c*l.n*l.size*l.size, sizeof(float));
    l.output_bn_backup = calloc(l.n*l.out_w*l.out_h, sizeof(float));
    l.biases_bn_backup = calloc(l.n, sizeof(float));
    if(l.layer_quant_flag && !l.close_quantization){
        l.forward = forward_convolutional_layer_quant_inputi_outputi_mkl;
    }else if (l.layer_quant_flag && l.close_quantization){
        l.forward = forward_convolutional_layer_quant_inputf_outputf;
    }else{
        l.forward = forward_convolutional_layer_nobn;
    }
#else
    l.forward = forward_convolutional_layer;
#endif
    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.output_int32 = calloc(l.batch*l.outputs, sizeof(int32_t));
    l.output_uint8_final = calloc(l.batch*l.outputs, sizeof(uint8_t));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(l.xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
#ifndef QUANTIZATION
    l.forward_gpu = forward_convolutional_layer_gpu;
#else
    l.min_activ_value_gpu = cuda_make_array(l.min_activ_value, 1);
    l.min_activ_value_gpu = cuda_make_array(l.min_activ_value, 1);

    l.activ_data_uint8_scales_gpu = cuda_make_array(l.activ_data_uint8_scales, 1);
    l.weight_data_uint8_scales_gpu = cuda_make_array(l.weight_data_uint8_scales, 1);
    l.biases_data_uint8_scales_gpu = cuda_make_array(l.biases_data_uint8_scales, l.c*l.n);
    l.output_data_uint8_scales_gpu = cuda_make_array(l.weight_data_uint8_scales, l.c*l.n);

    l.weights_quant_gpu = cuda_make_array(l.weights_uint8, l.c*l.n*l.size*l.size);
    l.weights_norm_gpu = cuda_make_array(l.weights_norm, l.nweights);

    l.weights_bn_backup_gpu = cuda_make_array(l.weights_bn_backup, l.c*l.n*l.size*l.size);
    l.biases_bn_backup_gpu = cuda_make_array(l.biases_bn_backup, l.n);
    l.output_bn_backup_gpu = cuda_make_array(l.output_bn_backup, l.n*l.out_w*l.out_h);

    l.forward_gpu = forward_convolutional_layer_quant_gpu;
#endif
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(l.xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    printf("conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
	l->output_bn_backup = realloc(l->output_bn_backup, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->output_bn_backup_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
	l->output_bn_backup_gpu = cuda_make_array(l->output_bn_backup, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

#ifdef QUANTIZATION
void forward_convolutional_layer_quant_inputi_outputi_mkl(convolutional_layer l, network net)
{
    int i, j, s, t;
    int batch_index, groups_index;
    // y = conv(x) --> q1*q2
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_h*l.out_w;
    net.workspace_quant16 = calloc(1, l.out_h*l.out_w*l.size*l.size*l.c*sizeof(int16_t));
    for(int in = 0; in < l.out_h*l.out_w*l.size*l.size*l.c; ++in){
        net.workspace_quant16[in] = l.input_data_uint8_zero_point[0];
    }
    if(l.count > 0){
        for (int input_index = 0; input_index < l.c*l.w*l.h; ++input_index) {
            l.input_int16[input_index] = (int16_t)net.input_uint8[input_index];
        }
    }
    for(batch_index = 0;batch_index < l.batch; batch_index++){
        for(groups_index = 0;groups_index < l.groups; groups_index++){
            int16_t *a16 = l.weights_int16 + groups_index*l.nweights/l.groups;
            int16_t *b16 = net.workspace_quant16;
            int32_t *c = l.output_int32 + (batch_index*l.groups + groups_index)*n*m;
            int16_t *im16 =  l.input_int16 + (batch_index*l.groups + groups_index)*l.c/l.groups*l.h*l.w;
            if (l.size == 1) {
                b16 = im16;
            } else {
                im2col_cpu_int16(im16, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b16, l.input_data_uint8_zero_point[0]);    // here
            }
            int co = 0;
            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, 
                                CblasNoTrans, CblasFixOffset, 
                                m, n, k, 
                                1, a16, k, 0,
                                b16, n, 0, 1, 
                                c, n, &co);

            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, 
                                CblasNoTrans, CblasFixOffset, 
                                m, n, k, 
                                -1, l.zero_point_int16, k, 0,
                                b16, n, 0, 1, 
                                c, n, &co);
        }
	}
    // y_i = alpha1 * conv(x) --> M*(nz1z2-z1a2-z2a1+q1q2) + z3
    for (i = 0; i < l.out_c; ++i) {
        for (j = 0; j < l.out_w*l.out_h; ++j){
            int out_index = i*l.out_w*l.out_h + j;
            int32_t output_quant_value = l.output_int32[out_index];
            int64_t temp_64bit = (output_quant_value + l.biases_int32[i])*l.M_value[i];
            output_quant_value = temp_64bit*l.M0_right_shift_value[i];
            switch (l.activation)
            {
            case LEAKY:
                output_quant_value = output_quant_value < 0 ? (round(output_quant_value*0.1) + l.activ_data_uint8_zero_point[0]): (output_quant_value + l.activ_data_uint8_zero_point[0]); 
                break;
            case LINEAR:
                output_quant_value = output_quant_value + l.activ_data_uint8_zero_point[0]; 
                break;
            case RELU:
                output_quant_value = output_quant_value < l.active_limit ? l.activ_data_uint8_zero_point[0]: (output_quant_value + l.activ_data_uint8_zero_point[0]); 
                break;
            default:
                break;
            }
            l.output_uint8_final[out_index] = clamp(output_quant_value, QUANT_NEGATIVE_LIMIT, QUANT_POSITIVE_LIMIT);
        }
    }
    if(l.quant_stop_flag){
        for (s = 0; s < l.out_c; ++s) {
            for (t = 0; t < l.out_w*l.out_h; ++t){
                int out_index = s*l.out_w*l.out_h + t;
                l.output[out_index] = (l.output_uint8_final[out_index] -  l.activ_data_uint8_zero_point[0]) * l.activ_data_uint8_scales[0];
            }
        }
    }
}

void forward_convolutional_layer_quant_inputi_outputi(convolutional_layer l, network net)
{
    int i, j, s, t;
    int batch_index, groups_index;
    // y = conv(x) --> q1*q2
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_h*l.out_w;
    net.workspace_quant = calloc(1, l.out_h*l.out_w*l.size*l.size*l.c*sizeof(uint8_t));
    for(int in = 0; in < l.out_h*l.out_w*l.size*l.size*l.c; ++in){
        net.workspace_quant[in] = l.input_data_uint8_zero_point[0];
    }
    for(batch_index = 0;batch_index < l.batch; batch_index++){
        for(groups_index = 0;groups_index < l.groups; groups_index++){
            uint8_t *a = l.weights_uint8 + groups_index*l.nweights/l.groups;
            uint8_t *b = (uint8_t *)net.workspace_quant;
            int32_t *c = l.output_int32 + (batch_index*l.groups + groups_index)*n*m;
            uint8_t *im =  net.input_uint8 + (batch_index*l.groups + groups_index)*l.c/l.groups*l.h*l.w;
            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu_uint8(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b, l.input_data_uint8_zero_point[0]);    // here
            }
            // #pragma omp parallel for
            for (int kk = 0; kk < l.out_c; ++kk){ 
                for (int ss = 0; ss < l.out_w*l.out_h; ++ss){
                    int index = kk * l.out_w*l.out_h + ss;
                    for (int tt = 0; tt < l.c*l.size*l.size; ++tt){
                        l.input_sum_int[index] += b[tt*l.out_w*l.out_h+ss];
                    }
                    l.input_sum_int[index] = l.input_sum_int[index] * l.weight_data_uint8_zero_point[kk];
                }
            }
            // 0.29s for whole net forward
            gemm_nn_uint8_int32_te(m, n, k, 1, a, k, b, n, 0, c, n);
            // cblas_gemm_s8u8s32(layout, transA, transB, offsetc, m, n, k, alpha,
            //         a, lda, ao, b, ldb, bo, beta, c, ldc, &co);
            gemm_nn_uint8_int32_te(m, n, k, -1, l.weight_data_uint8_zero_point, k, b, n, 1, c, n);
        }
	}
    // // y_i = alpha1 * conv(x) --> M*(nz1z2-z1a2-z2a1+q1q2) + z3
    #pragma omp parallel for
    for (i = 0; i < l.out_c; ++i) {
        for (j = 0; j < l.out_w*l.out_h; ++j){
            int out_index = i*l.out_w*l.out_h + j;
            // int32_t output_quant_value = l.output_int32[out_index] - l.input_sum_int[out_index];
            int32_t output_quant_value = l.output_int32[out_index];

            int64_t temp_64bit = (output_quant_value + l.biases_int32[i])*l.M_value[i];
            output_quant_value = temp_64bit*l.M0_right_shift_value[i];
            switch (l.activation)
            {
            case LEAKY:
                l.output_uint8_final[out_index] = output_quant_value < 0 ? (round(output_quant_value*0.1) + l.activ_data_uint8_zero_point[0]): (output_quant_value + l.activ_data_uint8_zero_point[0]); 
                break;
            case LINEAR:
            case RELU:
                l.output_uint8_final[out_index] = output_quant_value + l.activ_data_uint8_zero_point[0]; 
                break;
            default:
                break;
            }
            l.output_uint8_final[out_index] = clamp(l.output_uint8_final[out_index], QUANT_NEGATIVE_LIMIT, QUANT_POSITIVE_LIMIT);
        }
    }
    if(l.quant_stop_flag){
        #pragma omp parallel for
        for (s = 0; s < l.out_c; ++s) {
            for (t = 0; t < l.out_w*l.out_h; ++t){
                int out_index = s*l.out_w*l.out_h + t;
                l.output[out_index] = (l.output_uint8_final[out_index] -  l.activ_data_uint8_zero_point[0]) * l.activ_data_uint8_scales[0];
            }
        }
    }
}

void forward_convolutional_layer_quant_inputf_outputf(convolutional_layer l, network net)
{
    int i, j;
    int batch_index, groups_index;
    // y = conv(x) --> q1*q2
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_h*l.out_w;
    net.workspace_quant16 = calloc(1, l.out_h*l.out_w*l.size*l.size*l.c*sizeof(int16_t));
    for (int input_index = 0; input_index < l.c*l.w*l.h; ++input_index) {
        int16_t input_quant_value = round(net.input[input_index] / l.input_data_uint8_scales[0]) + l.input_data_uint8_zero_point[0];
        l.input_int16[input_index] = input_quant_value;
    }
    // for (int input_index = 0; input_index < l.c*l.w*l.h; ++input_index) {
    //     l.input_int16[input_index] = (int16_t)net.input_uint8[input_index];
    // }
    for(batch_index = 0;batch_index < l.batch; batch_index++){
        for(groups_index = 0;groups_index < l.groups; groups_index++){
            int16_t *a16 = l.weights_int16 + groups_index*l.nweights/l.groups;
            int16_t *b16 = (int16_t *)net.workspace_quant16;
            int32_t *c = l.output_int32 + (batch_index*l.groups + groups_index)*n*m;
            int16_t *im16 =  l.input_int16 + (batch_index*l.groups + groups_index)*l.c/l.groups*l.h*l.w;
            if (l.size == 1) {
                b16 = im16;
            } else {
                im2col_cpu_int16(im16, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b16, l.input_data_uint8_zero_point[0]);    // here
            }
            int co = 0;
            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, 
                                CblasNoTrans, CblasFixOffset, 
                                m, n, k, 
                                1, a16, k, 0,
                                b16, n, 0, 1, 
                                c, n, &co);

            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, 
                                CblasNoTrans, CblasFixOffset, 
                                m, n, k, 
                                -1, l.zero_point_int16, k, 0,
                                b16, n, 0, 1, 
                                c, n, &co);
        }
	}
    // dequant
    int num = 0, num_b = 0, temppp;
    printf("active limit = %d\n", l.active_limit);
    for (i = 0; i < l.n; ++i) {
        for (j = 0; j < l.out_w*l.out_h; ++j){
            int out_index = i*l.out_w*l.out_h + j;
            // int64_t output_quant_value = l.output_int32[out_index] + l.biases_int32[i];
            // l.output[out_index] = output_quant_value * rescale;
            int64_t temp_64bit = (l.output_int32[out_index] + l.biases_int32[i])*l.M_value[i];
            int32_t output_quant_value = temp_64bit*l.M0_right_shift_value[i];            
            switch (l.activation)
            {
            case LEAKY:
                temppp = output_quant_value < 0 ? (round(output_quant_value*0.1) + l.activ_data_uint8_zero_point[0]): (output_quant_value + l.activ_data_uint8_zero_point[0]);
                if(temppp < 0){
                    num_b++;  
                }
                l.output_uint8_final[out_index] = clamp(temppp, QUANT_NEGATIVE_LIMIT, QUANT_POSITIVE_LIMIT);
                break;
            case LINEAR:
                if(output_quant_value + l.activ_data_uint8_zero_point[0] < 0){
                    num_b++;  
                }
                temppp = output_quant_value + l.activ_data_uint8_zero_point[0]; 
                l.output_uint8_final[out_index] = clamp(temppp, QUANT_NEGATIVE_LIMIT, QUANT_POSITIVE_LIMIT);
                break;
            case RELU:
                output_quant_value = output_quant_value < 0 ? l.activ_data_uint8_zero_point[0]: (output_quant_value + l.activ_data_uint8_zero_point[0]); 
                break;
            default:
                printf("rong way!!!!\n");
                break;
            }
            int output_temp = (l.output_uint8_final[out_index] -  l.activ_data_uint8_zero_point[0]) * l.activ_data_uint8_scales[0];
            if(output_temp < 0 || output_temp > 255){
            // if(output_temp < 0){
                num++;
            }
            l.output[out_index] = (l.output_uint8_final[out_index] -  l.activ_data_uint8_zero_point[0]) * l.activ_data_uint8_scales[0];
        }
    }
    printf("layer %d, invalid uint8 num is %d\n",l.count ,num );
}
#endif

void forward_convolutional_layer_nobn(convolutional_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

    add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
}


void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }
	
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
	
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    }else{
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
    //     char file_name[100];
    // sprintf(file_name, "testcpu/%dcpu.txt", l.count);
    // FILE *fp = fopen(file_name, "w+");
    // for(int ii = 0; ii < l.outputs; ++ii){
    //     // printf("layer: %d, num = %d\n", l.count, l.outputs);
    //     fprintf(fp,"%f\n", l.output[ii]);
    // }
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return float_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

