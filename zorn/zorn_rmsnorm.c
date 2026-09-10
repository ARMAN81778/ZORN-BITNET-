
#include "zorn.h"
#include <math.h>

void zorn_rmsnorm(const float *x, const float *weight, float *out, int size, double eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < size; i++) sum_sq += (double)x[i] * (double)x[i];
    double mean_sq = sum_sq / (double)size;
    double inv_rms = 1.0 / sqrt(mean_sq + eps);

    for (int i = 0; i < size; i++) {
        out[i] = (float)((double)x[i] * inv_rms) * weight[i];
    }
}
