
#include "zorn.h"
#include <math.h>
#include <stdlib.h>

void zorn_rope_apply(float *vec, int num_heads, int head_dim, int position, double theta) {
    int half = head_dim / 2;
    double *cos_cache = malloc((size_t)half * sizeof(double));
    double *sin_cache = malloc((size_t)half * sizeof(double));
    if (!cos_cache || !sin_cache) {
        free(cos_cache); free(sin_cache);
        return;
    }

    for (int i = 0; i < half; i++) {
        double freq = pow(theta, -2.0 * (double)i / (double)head_dim);
        double angle = (double)position * freq;
        cos_cache[i] = cos(angle);
        sin_cache[i] = sin(angle);
    }

    for (int h = 0; h < num_heads; h++) {
        float *head_vec = vec + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float x0 = head_vec[i];
            float x1 = head_vec[i + half];
            double c = cos_cache[i], s = sin_cache[i];
            head_vec[i]        = (float)((double)x0 * c - (double)x1 * s);
            head_vec[i + half] = (float)((double)x0 * s + (double)x1 * c);
        }
    }

    free(cos_cache);
    free(sin_cache);
}
