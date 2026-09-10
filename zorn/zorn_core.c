
#include "zorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void print_zorn_banner(void) {
    printf("\033[1;35m");
    printf("████████╗ ██████╗ ██████╗ ███╗   ██╗\n");
    printf("╚══███╔══╝██╔═══██╗██╔══██╗████╗  ██║\n");
    printf("  ███╔╝   ██║   ██║██████╔╝██╔██╗ ██║\n");
    printf(" ███╔╝    ██║   ██║██╔══██╗██║╚██╗██║\n");
    printf("████████╗ ╚██████╔╝██║  ██║██║ ╚████║\n");
    printf("╚═══════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝\n");
    printf("   ⚡ ZORN native ternary engine (bitnet.cpp-style) ⚡\n");
    printf("   full transformer forward pass | GQA | RoPE | KV-cache | SIMD\n");
    printf("\033[0m");
}

static void zorn_load_model_config_json(ZORN_Engine *engine, const char *path) {
    strncpy(engine->model_name, "ZORN-Falcon3-Ternary", ZORN_MAX_MODEL_NAME - 1);
    engine->total_parameters = 0;

    FILE *file = fopen(path, "r");
    if (!file) return;

    char line[512];
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "model_name")) {
            sscanf(line, " \"model_name\": \"%127[^\"]\"", engine->model_name);
        } else if (strstr(line, "total_parameters")) {
            sscanf(line, " \"total_parameters\": %lld,", &engine->total_parameters);
        }
    }
    fclose(file);
}

int zorn_lm_head_load(ZORN_LMHead *head, const char *path, size_t vocab_size, size_t hidden_size) {
    memset(head, 0, sizeof(*head));
    head->fd = -1;
    size_t size = 0;
    int is_mmap = 0;
    void *data = zorn_mmap_readonly(path, &size, &is_mmap);
    size_t expected = vocab_size * hidden_size * sizeof(float);
    if (!data) {
        fprintf(stderr, "[zorn_lm_head] cannot read '%s'\n", path);
        return -1;
    }
    if (size < expected) {
        fprintf(stderr, "[zorn_lm_head] '%s' too small: %zu bytes, expected %zu\n",
                path, size, expected);
        zorn_unmap_or_free(data, size, is_mmap);
        return -1;
    }
    head->weight = (float *)data;
    head->mmap_base = data;
    head->mmap_size = size;
    head->is_mmap = is_mmap;
    head->vocab_size = vocab_size;
    head->hidden_size = hidden_size;
    head->loaded = 1;
    return 0;
}

void zorn_lm_head_free(ZORN_LMHead *head) {
    if (!head) return;
    zorn_unmap_or_free(head->mmap_base, head->mmap_size, head->is_mmap);
    memset(head, 0, sizeof(*head));
}

static float *zorn_load_final_norm(const char *model_dir, int hidden_size) {
    char path[ZORN_MAX_PATH];
    snprintf(path, sizeof(path), "%s/final_norm.bin", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[zorn_core] cannot open '%s'\n", path);
        return NULL;
    }
    float *buf = calloc((size_t)hidden_size, sizeof(float));
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, sizeof(float), (size_t)hidden_size, f);
    fclose(f);
    if ((int)rd != hidden_size) {
        fprintf(stderr, "[zorn_core] error: final_norm.bin size mismatch (got %zu floats, expected %d)\n", rd, hidden_size);
        free(buf);
        return NULL;
    }
    return buf;
}

int zorn_engine_init(ZORN_Engine *engine, const char *model_dir, int cpu_cores) {
    memset(engine, 0, sizeof(*engine));
    engine->config = zorn_default_config();
    strncpy(engine->model_path, model_dir, ZORN_MAX_PATH - 1);
    engine->num_cores_to_use = (cpu_cores > 0) ? cpu_cores : 1;

    char config_path[ZORN_MAX_PATH];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    zorn_load_model_config_json(engine, config_path);

    
    int simd_level = zorn_simd_detect_level();
    engine->metrics.simd_level_used = simd_level;
    fprintf(stderr, "[zorn_core] SIMD level detected: %s\n", zorn_simd_level_name(simd_level));

    
    char tokenizer_path[ZORN_MAX_PATH];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", model_dir);
    int tok_ok = (zorn_tokenizer_load(&engine->tokenizer, tokenizer_path) == 0);

    
    int layers_ok = (zorn_layers_load(engine, model_dir) == 0);

    if (engine->config.hidden_size <= 0 || engine->config.num_layers < 0 ||
        engine->config.vocab_size <= 0 || engine->config.max_context_length <= 0 ||
        engine->config.num_attn_heads <= 0 || engine->config.num_kv_heads <= 0 ||
        engine->config.head_dim <= 0 ||
        engine->config.num_attn_heads % engine->config.num_kv_heads != 0) {
        fprintf(stderr, "[zorn_core] invalid model configuration\n");
        layers_ok = 0;
    }

    int hidden_size = engine->config.hidden_size;
    size_t vocab_size = tok_ok ? engine->tokenizer.vocab_size : (size_t)engine->config.vocab_size;

    
    char emb_path[ZORN_MAX_PATH];
    snprintf(emb_path, sizeof(emb_path), "%s/embedding.bin", model_dir);
    int emb_ok = (zorn_embedding_load(&engine->embedding, emb_path, vocab_size, (size_t)hidden_size) == 0);

    
    char lm_path[ZORN_MAX_PATH];
    snprintf(lm_path, sizeof(lm_path), "%s/lm_head.bin", model_dir);
    int lm_ok = (zorn_lm_head_load(&engine->lm_head, lm_path, vocab_size, (size_t)hidden_size) == 0);

    
    engine->final_norm_weight = zorn_load_final_norm(model_dir, hidden_size);
    int final_norm_ok = (engine->final_norm_weight != NULL);

    
    int kv_ok = (zorn_kv_cache_init(engine) == 0);

    int vocab_ok = tok_ok && engine->tokenizer.vocab_size == (size_t)engine->config.vocab_size;
    engine->real_generation_ready = vocab_ok && layers_ok && emb_ok && lm_ok && final_norm_ok && kv_ok;

    if (engine->real_generation_ready) {
        fprintf(stderr, "[zorn_core] FULL forward-pass generation ENABLED "
                "(%d/%d layers, vocab=%zu, hidden=%d, SIMD=%s)\n",
                engine->layers_loaded, engine->config.num_layers, vocab_size,
                hidden_size, zorn_simd_level_name(simd_level));
    } else {
        fprintf(stderr, "[zorn_core] FULL generation DISABLED - missing one of: "
                "tokenizer=%d layers=%d(%d/%d) embedding=%d lm_head=%d final_norm=%d kv_cache=%d\n",
                tok_ok && vocab_ok, layers_ok, engine->layers_loaded, engine->config.num_layers,
                emb_ok, lm_ok, final_norm_ok, kv_ok);
    }

    double weights_mb = (double)engine->shared_weights_mmap_size / (1024.0 * 1024.0);
    double emb_mb = emb_ok ? (double)engine->embedding.mmap_size / (1024.0 * 1024.0) : 0.0;
    double lm_mb = lm_ok ? (double)engine->lm_head.mmap_size / (1024.0 * 1024.0) : 0.0;
    engine->metrics.ram_usage_mb = weights_mb + emb_mb + lm_mb + 200.0;

    engine->kb.docs = NULL;
    return 0; 
}

void zorn_engine_free(ZORN_Engine *engine) {
    if (!engine) return;
    zorn_layers_free(engine);
    zorn_unmap_or_free(engine->shared_weights_mmap_base, engine->shared_weights_mmap_size,
                        engine->shared_weights_is_mmap);
    engine->shared_weights_mmap_base = NULL;
    zorn_embedding_free(&engine->embedding);
    zorn_lm_head_free(&engine->lm_head);
    free(engine->final_norm_weight);
    zorn_kv_cache_free(engine);
    zorn_kb_free(&engine->kb);
    zorn_tokenizer_free(&engine->tokenizer);
    zorn_log_close(engine);
}



size_t zorn_sample_from_logits(const float *logits, size_t vocab_size, double temperature,
                                unsigned int *rng_state) {
    double max_logit = -1e300;
    for (size_t v = 0; v < vocab_size; v++)
        if (logits[v] > max_logit) max_logit = logits[v];

    double *probs = malloc(vocab_size * sizeof(double));
    if (!probs) return vocab_size - 1;
    double sum_exp = 0.0;
    double temp = temperature > 1e-6 ? temperature : 1e-6;
    for (size_t v = 0; v < vocab_size; v++) {
        probs[v] = exp(((double)logits[v] - max_logit) / temp);
        sum_exp += probs[v];
    }

    double r = (double)zorn_rand_next(rng_state) / 4294967296.0;
    double cumulative = 0.0;
    size_t chosen = vocab_size - 1;
    for (size_t v = 0; v < vocab_size; v++) {
        cumulative += probs[v] / sum_exp;
        if (r <= cumulative) { chosen = v; break; }
    }
    free(probs);
    return chosen;
}



int zorn_log_open(ZORN_Engine *engine, const char *log_path) {
    strncpy(engine->log_path, log_path, ZORN_MAX_PATH - 1);
    engine->log_fp = fopen(log_path, "a");
    if (!engine->log_fp) {
        fprintf(stderr, "[zorn_core] cannot open log file '%s'\n", log_path);
        return -1;
    }
    return 0;
}

static void zorn_json_escape(const char *in, char *out, size_t out_size) {
    size_t oi = 0;
    for (size_t i = 0; in[i] != '\0' && oi + 2 < out_size; i++) {
        char c = in[i];
        if (c == '"' || c == '\\') { out[oi++] = '\\'; out[oi++] = c; }
        else if (c == '\n') { out[oi++] = '\\'; out[oi++] = 'n'; }
        else out[oi++] = c;
    }
    out[oi] = '\0';
}

void zorn_log_turn(ZORN_Engine *engine, const char *prompt,
                    ZORN_Document **retrieved, int retrieved_count, const char *response) {
    if (!engine->log_fp) return;

    time_t now = time(NULL);
    char esc_prompt[ZORN_LOG_LINE_MAX], esc_response[ZORN_LOG_LINE_MAX];
    zorn_json_escape(prompt, esc_prompt, sizeof(esc_prompt));
    zorn_json_escape(response, esc_response, sizeof(esc_response));

    fprintf(engine->log_fp, "{\"ts\":%ld,\"prompt\":\"%s\",\"response\":\"%s\",",
            (long)now, esc_prompt, esc_response);

    fprintf(engine->log_fp, "\"retrieved\":[");
    for (int i = 0; i < retrieved_count; i++) {
        char esc_src[256];
        zorn_json_escape(retrieved[i]->source_path, esc_src, sizeof(esc_src));
        fprintf(engine->log_fp, "%s{\"source\":\"%s\",\"len\":%zu}",
                i > 0 ? "," : "", esc_src, retrieved[i]->content_len);
    }
    fprintf(engine->log_fp, "],");

    fprintf(engine->log_fp,
            "\"metrics\":{\"ttft_s\":%.4f,\"tokens_per_sec\":%.2f,\"tokens\":%ld,"
            "\"ram_mb\":%.2f,\"simd\":\"%s\",\"real_gen\":%s}}\n",
            engine->metrics.time_to_first_token, engine->metrics.tokens_per_sec,
            engine->metrics.total_tokens_generated, engine->metrics.ram_usage_mb,
            zorn_simd_level_name((int)engine->metrics.simd_level_used),
            engine->real_generation_ready ? "true" : "false");

    fflush(engine->log_fp);
}

void zorn_log_close(ZORN_Engine *engine) {
    if (engine->log_fp) { fclose(engine->log_fp); engine->log_fp = NULL; }
}



void zorn_execute_inference(ZORN_Engine *engine, const char *prompt) {
    double start = zorn_time_seconds(), first_token = 0.0, end = 0.0;

    ZORN_Document *retrieved[ZORN_RAG_TOPK_DEFAULT];
    float scores[ZORN_RAG_TOPK_DEFAULT];
    int n_retrieved = 0;
    if (engine->kb.count > 0) {
        n_retrieved = zorn_kb_retrieve_topk(&engine->kb, prompt, ZORN_RAG_TOPK_DEFAULT, retrieved, scores);
    }

    for (int i = 0; i < n_retrieved; i++) {
        printf("   [%.3f] %s: %.80s...\n", scores[i], retrieved[i]->source_path, retrieved[i]->content);
    }

    printf("\n🤖 [\033[1;32m%s\033[0m]: ", engine->model_name);
    fflush(stdout);

    char response_buf[ZORN_LOG_LINE_MAX] = {0};
    engine->metrics.total_tokens_generated = 0;

    if (!engine->real_generation_ready) {
        printf("ZORN");
        end = zorn_time_seconds();
        zorn_log_turn(engine, prompt, retrieved, n_retrieved, response_buf);
        return;
    }

    
    unsigned int rng_state = (unsigned int)time(NULL) ^ (unsigned int)(size_t)prompt;
    const int max_new_tokens = 60;

    
    int position = engine->kv_cache[0].current_len;
    size_t next_token_id = 0; 

    for (int i = 0; i < max_new_tokens; i++) {
        double t0 = 0.0;
        double s0 = zorn_time_seconds();

        float *logits = zorn_transformer_forward_token(engine, next_token_id, position, engine->num_cores_to_use);

        double s1 = zorn_time_seconds();
        t0 = (s1 - s0) * 1000.0;
        engine->metrics.last_matmul_ms = t0;

        size_t chosen = zorn_sample_from_logits(logits, engine->tokenizer.vocab_size, 0.8, &rng_state);
        free(logits);

        const char *piece = zorn_tokenizer_decode_id(&engine->tokenizer, chosen);

        if (i == 0) {
            first_token = zorn_time_seconds();
            engine->metrics.time_to_first_token = first_token - start;
        }

        printf("%s", piece);
        fflush(stdout);
        strncat(response_buf, piece, sizeof(response_buf) - strlen(response_buf) - 1);
        engine->metrics.total_tokens_generated++;

        if (strstr(piece, "<|endoftext|>") || strstr(piece, "</s>")) break;

        next_token_id = chosen;
        position++;
        if (position >= engine->config.max_context_length) break;
    }
    printf("\n");

    end = zorn_time_seconds();
    double total_sec = end - start;
    engine->metrics.tokens_per_sec = (double)engine->metrics.total_tokens_generated / total_sec;

    printf("\n📊 [METRICS]: TTFT: \033[1;36m%.4f s\033[0m | Speed: \033[1;36m%.2f tok/s\033[0m | "
           "RAM: %.2f MB | SIMD: %s | forward/token: %.2f ms\n",
           engine->metrics.time_to_first_token, engine->metrics.tokens_per_sec,
           engine->metrics.ram_usage_mb, zorn_simd_level_name((int)engine->metrics.simd_level_used),
           engine->metrics.last_matmul_ms);
    printf("──────────────────────────────────────────────────────\n");

    zorn_log_turn(engine, prompt, retrieved, n_retrieved, response_buf);
}
