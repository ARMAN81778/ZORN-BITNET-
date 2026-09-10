
#include "zorn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int zorn_kv_cache_reserve(ZORN_KVCacheLayer *cache, int kv_dim, int required) {
    if (!cache || required <= cache->capacity) return 0;
    int new_cap = cache->capacity > 0 ? cache->capacity : 256;
    while (new_cap < required) {
        int next = new_cap > 1073741823 ? required : new_cap * 2;
        if (next <= new_cap) { new_cap = required; break; }
        new_cap = next;
    }
    if (new_cap > required && required <= 32768) new_cap = required;
    size_t elems = (size_t)new_cap * (size_t)kv_dim;
    float *new_k = calloc(elems, sizeof(float));
    float *new_v = calloc(elems, sizeof(float));
    if (!new_k || !new_v) {
        free(new_k); free(new_v);
        return -1;
    }
    if (cache->k_cache && cache->capacity > 0) {
        size_t old_elems = (size_t)cache->capacity * (size_t)kv_dim;
        memcpy(new_k, cache->k_cache, old_elems * sizeof(float));
        memcpy(new_v, cache->v_cache, old_elems * sizeof(float));
    }
    free(cache->k_cache);
    free(cache->v_cache);
    cache->k_cache = new_k;
    cache->v_cache = new_v;
    cache->capacity = new_cap;
    return 0;
}

void zorn_attention_forward(ZORN_Engine *engine, int layer_idx, const float *hidden_in,
                             float *hidden_out, int position, int num_threads) {
    ZORN_ModelConfig *cfg = &engine->config;
    ZORN_TransformerLayer *layer = &engine->layers[layer_idx];
    ZORN_KVCacheLayer *cache = &engine->kv_cache[layer_idx];

    int q_dim = cfg->q_dim;
    int kv_dim = cfg->kv_dim;
    int num_heads = cfg->num_attn_heads;
    int num_kv_heads = cfg->num_kv_heads;
    int head_dim = cfg->head_dim;
    int heads_per_group = num_heads / num_kv_heads;

    float *q = malloc((size_t)q_dim * sizeof(float));
    float *k = malloc((size_t)kv_dim * sizeof(float));
    float *v = malloc((size_t)kv_dim * sizeof(float));
    if (!q || !k || !v) {
        free(q); free(k); free(v);
        memset(hidden_out, 0, (size_t)cfg->hidden_size * sizeof(float));
        return;
    }

    zorn_ternary_matvec(&layer->q_proj, hidden_in, q, num_threads);
    zorn_ternary_matvec(&layer->k_proj, hidden_in, k, num_threads);
    zorn_ternary_matvec(&layer->v_proj, hidden_in, v, num_threads);

    
    zorn_rope_apply(q, num_heads, head_dim, position, cfg->rope_theta);
    zorn_rope_apply(k, num_kv_heads, head_dim, position, cfg->rope_theta);

    
    if (position < 0 || position >= cfg->max_context_length ||
        zorn_kv_cache_reserve(cache, kv_dim, position + 1) != 0) {
        memset(hidden_out, 0, (size_t)cfg->hidden_size * sizeof(float));
        free(q); free(k); free(v);
        return;
    }

    {
        memcpy(cache->k_cache + (size_t)position * kv_dim, k, (size_t)kv_dim * sizeof(float));
        memcpy(cache->v_cache + (size_t)position * kv_dim, v, (size_t)kv_dim * sizeof(float));
        if (position + 1 > cache->current_len) cache->current_len = position + 1;
    }

    int seq_len = cache->current_len;
    float *attn_out = calloc((size_t)q_dim, sizeof(float));
    double scale = 1.0 / sqrt((double)head_dim);

    
    float *scores = malloc((size_t)seq_len * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int kv_head = h / heads_per_group;
        const float *q_head = q + (size_t)h * head_dim;

        double max_score = -1e300;
        for (int t = 0; t < seq_len; t++) {
            const float *k_t = cache->k_cache + (size_t)t * kv_dim + (size_t)kv_head * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; d++) dot += (double)q_head[d] * k_t[d];
            double s = dot * scale;
            scores[t] = (float)s;
            if (s > max_score) max_score = s;
        }

        double sum_exp = 0.0;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = (float)exp((double)scores[t] - max_score);
            sum_exp += scores[t];
        }
        if (!(sum_exp > 0.0) || !isfinite(sum_exp)) {
            free(scores); free(q); free(k); free(v); free(attn_out);
            memset(hidden_out, 0, (size_t)cfg->hidden_size * sizeof(float));
            return;
        }
        for (int t = 0; t < seq_len; t++) scores[t] = (float)((double)scores[t] / sum_exp);

        float *out_head = attn_out + (size_t)h * head_dim;
        for (int t = 0; t < seq_len; t++) {
            const float *v_t = cache->v_cache + (size_t)t * kv_dim + (size_t)kv_head * head_dim;
            float w = scores[t];
            for (int d = 0; d < head_dim; d++) out_head[d] += w * v_t[d];
        }
    }

    free(scores);

    
    zorn_ternary_matvec(&layer->o_proj, attn_out, hidden_out, num_threads);

    free(q); free(k); free(v); free(attn_out);
}
