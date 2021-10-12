#ifndef SPECIALMATH_CUDA_KERNELS_H
#define SPECIALMATH_CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void lgamma_cuda_launcher(cudaStream_t stream, float *input, float *output, int N);
void digamma_cuda_launcher(cudaStream_t stream, float *input, float *output, int N);

#ifdef __cplusplus
}
#endif

#endif