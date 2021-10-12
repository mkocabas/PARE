#include <THC/THC.h>
#include "specialmath_cuda_impl.h"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int lgamma_cuda(THCudaTensor *input, THCudaTensor *output)
{
    THCudaTensor_resizeAs(state, output, input);

    float *output_f = THCudaTensor_data(state, output);
    float *input_f = THCudaTensor_data(state, input);

    cudaStream_t stream = THCState_getCurrentStream(state);

    const int size = THCudaTensor_nElement(state, input);
    lgamma_cuda_launcher(stream, input_f, output_f, size);

    return 1;
}

int digamma_cuda(THCudaTensor *input, THCudaTensor *output)
{
    THCudaTensor_resizeAs(state, output, input);

    float *output_f = THCudaTensor_data(state, output);
    float *input_f = THCudaTensor_data(state, input);

    cudaStream_t stream = THCState_getCurrentStream(state);

    const int size = THCudaTensor_nElement(state, input);
    digamma_cuda_launcher(stream, input_f, output_f, size);

    return 1;
}
