#include "zorn.h"
#include <stdint.h>

float zorn_i2s_scale_for(const ZORN_TernaryMatrix *W, size_t weight_index) {
    if (!W) return 1.0f;
    if (!W->scales || W->group_size <= 0) return W->scale;
    size_t row_stride = (size_t)W->in_features;
    size_t row = row_stride ? weight_index / row_stride : 0;
    size_t col = row_stride ? weight_index % row_stride : 0;
    size_t groups_per_row = (size_t)W->num_groups;
    size_t g = groups_per_row ? col / (size_t)W->group_size : 0;
    size_t idx = row * groups_per_row + g;
    if (row >= (size_t)W->out_features || g >= groups_per_row) return W->scale;
    return W->scales[idx];
}
