#include <stdio.h>
#include <math.h>
#include <float.h>

#include "specialmath_cuda_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE 512
#define FALSE 0
#define TRUE 1
#define MAXNUM FLT_MAX

__global__ void lgamma_kernel(float *in, float *out, int N)
{
    for (int tid = blockIdx.x*blockDim.x + threadIdx.x; tid < N; tid += blockDim.x*gridDim.x) {
        out[tid] = lgammaf(in[tid]);
    }
}


__device__ float digammaf(float y)
{
    double x = (double) y;
    double neginf = -FLT_MAX;
    const double c = 12;
    const double digamma1 = -0.57721566490153286;
    const double trigamma1 = 1.6449340668482264365; /* pi^2/6 */
    const double s = 1e-6;
    const double s3 = 1./12;
    const double s4 = 1./120;
    const double s5 = 1./252;
    const double s6 = 1./240;
    const double s7 = 1./132;
    // const double s8 = 691./32760;
    // const double s9 = 1./12;
    // const double s10 = 3617./8160;
    double result;

      /* Singularities */
    if ((x <= 0) && (floor(x) == x)) {
        return neginf;
    }

    /* Use Taylor series if argument <= S */
    if (x <= s) {
        return digamma1 - 1/x + trigamma1*x;
    }

    /* Reduce to digamma(X + N) where (X + N) >= C */
    result = 0;
    while (x < c) {
        result -= 1/x;
        x++;
    }
    /* Use de Moivre's expansion if argument >= C */
    /* This expansion can be computed in Maple via asympt(Psi(x),x) */
    if (x >= c) {
        double r = 1/x;
        result += log(x) - 0.5*r;
        r *= r;
    #if 1
        result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
    #else
        /* this version for lame compilers */
        double t = (s5 - r * (s6 - r * s7));
        result -= r * (s3 - r * (s4 - r * t));
    #endif
    }

    /* assign the result to the pointer*/
    return (float) result;
}



__global__ void digamma_kernel(float *in, float *out, int N)
{
    for (int tid = blockIdx.x*blockDim.x + threadIdx.x; tid < N; tid += blockDim.x*gridDim.x) {
        out[tid] = digammaf(in[tid]);
    }
}


void check_errors()
{
    const cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void lgamma_cuda_launcher(cudaStream_t stream, float *input, float *output, int N)
{
    const int gridSize = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    lgamma_kernel<<<gridSize, BLOCKSIZE, 0, stream>>>(input, output, N);
    check_errors();
}

void digamma_cuda_launcher(cudaStream_t stream, float *input, float *output, int N)
{
    const int gridSize = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    digamma_kernel<<<gridSize, BLOCKSIZE, 0, stream>>>(input, output, N);
    check_errors();
}


#ifdef __cplusplus
}
#endif
