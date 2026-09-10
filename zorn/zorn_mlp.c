#include "zorn.h"
#include <stdlib.h>
#include <math.h>

static inline float zorn_silu(float x) {
    return x / (1.0f + expf(-x));
}

void zorn_mlp_forward(ZORN_Engine *engine, int layer_idx, const float *hidden_in,
                       float *hidden_out, int num_threads) {
    ZORN_ModelConfig *cfg = &engine->config;
    ZORN_TransformerLayer *layer = &engine->layers[layer_idx];
    int inter = cfg->mlp_intermediate;

    float *gate = malloc((size_t)inter * sizeof(float));
    float *up = malloc((size_t)inter * sizeof(float));
    float *fused = malloc((size_t)inter * sizeof(float));

    zorn_ternary_matvec(&layer->gate_proj, hidden_in, gate, num_threads);
    zorn_ternary_matvec(&layer->up_proj, hidden_in, up, num_threads);

    for (int i = 0; i < inter; i++) fused[i] = zorn_silu(gate[i]) * up[i];

    zorn_ternary_matvec(&layer->down_proj, fused, hidden_out, num_threads);

    free(gate); free(up); free(fused);
}
