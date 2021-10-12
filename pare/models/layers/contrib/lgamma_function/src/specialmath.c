#include <TH/TH.h>
#include <math.h>
#include <float.h>

#define SUCCESS 1
#define FAILURE 0
#define MAXNUM FLT_MAX
#define FALSE 0
#define TRUE 1


static const float A[] = {
    -4.16666666666666666667E-3f,
    3.96825396825396825397E-3f,
    -8.33333333333333333333E-3f,
    8.33333333333333333333E-2f
};

static inline float polevl3(float x)
{
    return ((A[0]*x + A[1])*x + A[2])*x + A[3];
}

static float digamma_impl_maybe_poly(float s)
{
    float z;
    if (s < 1.0e8f) {
        z = 1.0f / (s * s);
        return z * polevl3(z);
    }
    return 0.0f;
}

static float digamma_impl(float x)
{
    float p, q, nz, s, w, y;
    int negative = FALSE;

    const float maxnum = MAXNUM;
    const float m_pi = M_PI;

    const float zero = 0.0f;
    const float one = 1.0f;
    const float half = 0.5f;
    nz = zero;

    if (x <= zero) {
        negative = TRUE;
        q = x;
        p = floor(q);
        if (p == q) {
            return maxnum;
        }
        /* Remove the zeros of tan(m_pi x)
         * by subtracting the nearest integer from x
         */
        nz = q - p;
        if (nz != half) {
            if (nz > half) {
                p += one;
                nz = q - p;
            }
            nz = m_pi / tan(m_pi * nz);

        } else {
            nz = zero;
        }
        x = one - x;
    }

    /* use the recurrence psi(x+1) = psi(x) + 1/x. */
    s = x;
    w = zero;
    while (s < 10.0f) {
      w += one / s;
      s += one;
    }

    y = digamma_impl_maybe_poly(s);

    y = log(s) - (half / s) - y - w;
    return (negative == TRUE) ? y - nz : y;
}


// float digamma_impl(float y)
// {
//     double x = (double) y;
//     double neginf = -FLT_MAX;
//     const double c = 12;
//     const double digamma1 = -0.57721566490153286;
//     const double trigamma1 = 1.6449340668482264365; /* pi^2/6 */
//     const double s = 1e-6;
//     const double s3 = 1./12;
//     const double s4 = 1./120;
//     const double s5 = 1./252;
//     const double s6 = 1./240;
//     const double s7 = 1./132;
//     // const double s8 = 691./32760;
//     // const double s9 = 1./12;
//     // const double s10 = 3617./8160;
//     double result;

//       /* Singularities */
//     if ((x <= 0) && (floor(x) == x)) {
//         return neginf;
//     }

//     /* Use Taylor series if argument <= S */
//     if (x <= s) {
//         return digamma1 - 1/x + trigamma1*x;
//     }

//     /* Reduce to digamma(X + N) where (X + N) >= C */
//     result = 0;
//     while (x < c) {
//         result -= 1/x;
//         x++;
//     }
//     /* Use de Moivre's expansion if argument >= C */
//     /* This expansion can be computed in Maple via asympt(Psi(x),x) */
//     if (x >= c) {
//         double r = 1/x;
//         result += log(x) - 0.5*r;
//         r *= r;
//     #if 1
//         result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
//     #else
//         /* this version for lame compilers */
//         double t = (s5 - r * (s6 - r * s7));
//         result -= r * (s3 - r * (s4 - r * t));
//     #endif
//     }

//     /* assign the result to the pointer*/
//     return (float) result;
// }

//----------------------------------------------------------------------------
// log gamma function
//----------------------------------------------------------------------------
int lgamma_cpu(THFloatTensor *input, THFloatTensor *output)
{
    THFloatTensor_resizeAs(output, input);

    // Work on raw data
    const float *in = THFloatTensor_data(input);
    const int size = THFloatTensor_nElement(input);
    float *out = THFloatTensor_data(output);

    // Apply elementwise op
    for (int i = 0; i < size; ++i) {
        out[i] = lgamma(in[i]);
    }

    return SUCCESS;
}

//----------------------------------------------------------------------------
// digamma function
//----------------------------------------------------------------------------
int digamma_cpu(THFloatTensor *input, THFloatTensor *output)
{
    THFloatTensor_resizeAs(output, input);

    // Work on raw data
    const float *in = THFloatTensor_data(input);
    const int size = THFloatTensor_nElement(input);
    float *out = THFloatTensor_data(output);

    // Apply elementwise op
    for (int i = 0; i < size; ++i) {
        out[i] = digamma_impl(in[i]);
    }

    return SUCCESS;
}


