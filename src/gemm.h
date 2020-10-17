#ifndef GEMM_H
#define GEMM_H
#include "stdint.h"

void gemm_nn_int8_int16(int M, int N, int K, int8_t ALPHA,
    int8_t *A, int lda,
    int8_t *B, int ldb,
    int16_t *C, int ldc);

void gemm_nn_int8_int32(int M, int N, int K, int8_t ALPHA,
    int8_t *A, int lda,
    int8_t *B, int ldb,
    int32_t *C, int ldc);

void gemm_nn_uint8_int32(int M, int N, int K, float ALPHA, 
        uint8_t *A, int lda, 
        uint8_t *B, int ldb,
        int32_t *C, int ldc);

void gemm_nn_uint8_uint32(int M, int N, int K, float ALPHA, 
        uint8_t *A, int lda, 
        uint8_t *B, int ldb,
        uint32_t *C, int ldc);

void gemm_nn_uint8_uint32_register(int M, int N, int K, uint8_t ALPHA,
    uint8_t *A, int lda,
    uint8_t *B, int ldb,
    uint32_t *C, int ldc);

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif
#endif
