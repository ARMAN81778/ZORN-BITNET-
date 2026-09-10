
#include "zorn.h"
#include <stdlib.h>
#include <string.h>


static void zorn_lm_head_matvec(const ZORN_LMHead *head, const float *hidden, float *logits) {
    size_t V = head->vocab_size, H = head->hidden_size;
    for (size_t v = 0; v < V; v++) {
        const float *row = head->weight + v * H;
        float acc = 0.0f;
        for (size_t h = 0; h < H; h++) acc += row[h] * hidden[h];
        logits[v] = acc;
    }
}

float *zorn_transformer_forward_token(ZORN_Engine *engine, size_t token_id, int position, int num_threads) {
    ZORN_ModelConfig *cfg = &engine->config;
    int hidden_size = cfg->hidden_size;

    float *hidden = malloc((size_t)hidden_size * sizeof(float));
    float *normed = malloc((size_t)hidden_size * sizeof(float));
    float *sub_out = malloc((size_t)hidden_size * sizeof(float));

    if (token_id < engine->embedding.vocab_size) {
        memcpy(hidden, engine->embedding.weight + token_id * (size_t)hidden_size,
               (size_t)hidden_size * sizeof(float));
    } else {
        memset(hidden, 0, (size_t)hidden_size * sizeof(float));
    }

    for (int li = 0; li < cfg->num_layers; li++) {
        ZORN_TransformerLayer *L = &engine->layers[li];

        
        zorn_rmsnorm(hidden, L->input_layernorm_weight, normed, hidden_size, cfg->rmsnorm_eps);
        zorn_attention_forward(engine, li, normed, sub_out, position, num_threads);
        for (int i = 0; i < hidden_size; i++) hidden[i] += sub_out[i]; /* residual 1 */

        
        zorn_rmsnorm(hidden, L->post_attention_layernorm_weight, normed, hidden_size, cfg->rmsnorm_eps);
        zorn_mlp_forward(engine, li, normed, sub_out, num_threads);
        for (int i = 0; i < hidden_size; i++) hidden[i] += sub_out[i]; /* residual 2 */
    }

    zorn_rmsnorm(hidden, engine->final_norm_weight, normed, hidden_size, cfg->rmsnorm_eps);

    
    float *logits = malloc((size_t)cfg->vocab_size * sizeof(float));
    zorn_lm_head_matvec(&engine->lm_head, normed, logits);

    free(hidden); free(normed); free(sub_out);
    return logits;
}
