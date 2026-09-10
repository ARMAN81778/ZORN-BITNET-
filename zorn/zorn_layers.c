
#include "zorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



static char *zorn_read_whole_file_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}


static double zorn_json_find_number(const char *text, const char *key, double fallback) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    char *pos = strstr(text, pattern);
    if (!pos) return fallback;
    pos = strchr(pos, ':');
    if (!pos) return fallback;
    return strtod(pos + 1, NULL);
}


static int zorn_json_find_string(const char *text, const char *key, char *out, size_t out_size) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    char *pos = strstr(text, pattern);
    if (!pos) return -1;
    pos = strchr(pos, ':');
    if (!pos) return -1;
    pos = strchr(pos, '"');
    if (!pos) return -1;
    pos++;
    char *end = strchr(pos, '"');
    if (!end) return -1;
    size_t len = (size_t)(end - pos);
    if (len >= out_size) len = out_size - 1;
    memcpy(out, pos, len);
    out[len] = '\0';
    return 0;
}


static char *zorn_json_find_block(const char *text, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    char *pos = strstr(text, pattern);
    if (!pos) return NULL;
    pos = strchr(pos, ':');
    if (!pos) return NULL;
    pos = strchr(pos, '{');
    return pos;
}


static size_t zorn_json_block_len(const char *block_start) {
    int depth = 0;
    const char *p = block_start;
    do {
        if (*p == '{') depth++;
        else if (*p == '}') depth--;
        p++;
    } while (*p && depth > 0);
    return (size_t)(p - block_start);
}


static int zorn_parse_ternary_field(const char *layer_block, size_t layer_block_len,
                                     const char *field_name, ZORN_TernaryMatrix *out,
                                     uint8_t *shared_mmap_base, size_t shared_mmap_size,
                                     const char *scales_path) {
    char *tmp = malloc(layer_block_len + 1);
    memcpy(tmp, layer_block, layer_block_len);
    tmp[layer_block_len] = '\0';

    char *field_block = zorn_json_find_block(tmp, field_name);
    if (!field_block) { free(tmp); return -1; }

    double byte_offset = zorn_json_find_number(field_block, "byte_offset", -1);
    double out_features = zorn_json_find_number(field_block, "out_features", -1);
    double in_features = zorn_json_find_number(field_block, "in_features", -1);
    double scale = zorn_json_find_number(field_block, "scale", 1.0);
    double scale_offset = zorn_json_find_number(field_block, "scale_offset", -1);
    double group_size = zorn_json_find_number(field_block, "group_size", 0);
    double num_groups = zorn_json_find_number(field_block, "num_groups", 0);

    free(tmp);

    if (byte_offset < 0 || out_features < 0 || in_features < 0) return -1;

    out->out_features = (int)out_features;
    out->in_features = (int)in_features;
    out->scale = (float)scale;
    out->scales = NULL;
    out->group_size = (int)group_size;
    out->num_groups = (int)num_groups;
    out->scale_mmap_base = NULL;
    out->scale_mmap_size = 0;

    if (scale_offset >= 0 && out->group_size > 0 && out->num_groups > 0 && scales_path) {
        FILE *sf = fopen(scales_path, "rb");
        if (sf) {
            if (fseek(sf, (long)scale_offset, SEEK_SET) == 0) {
                size_t bytes = (size_t)out->out_features * (size_t)out->num_groups * sizeof(float);
                out->scales = (float *)malloc(bytes);
                if (!out->scales || fread(out->scales, 1, bytes, sf) != bytes) {
                    free(out->scales); out->scales = NULL;
                }
            }
            fclose(sf);
        }
        if (!out->scales) {
            fprintf(stderr, "[zorn_layers] error: could not load group scales for '%s'\n", field_name);
            return -1;
        }
    }

    size_t needed_weights = (size_t)out_features * (size_t)in_features;
    size_t needed_bytes = (needed_weights + 3) / 4;

    out->block.packed_data = shared_mmap_base + (size_t)byte_offset;
    out->block.packed_size = needed_bytes;
    out->block.num_weights = needed_weights;
    out->block.is_mmap = 0;   
    out->block.mmap_base = NULL;
    out->block.mmap_size = 0;
    out->block.fd = -1;

    if ((size_t)byte_offset + needed_bytes > shared_mmap_size) {
        fprintf(stderr, "[zorn_layers] warning: field '%s' offset+size exceeds weights.zrn bounds\n", field_name);
        return -1;
    }
    return 0;
}


static float *zorn_load_fp32_vector(const char *dir, const char *filename, int expected_size) {
    if (!filename || filename[0] == '\0') return NULL;
    char path[ZORN_MAX_PATH];
    snprintf(path, sizeof(path), "%s/%s", dir, filename);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[zorn_layers] cannot open fp32 vector '%s'\n", path);
        return NULL;
    }
    float *buf = malloc((size_t)expected_size * sizeof(float));
    size_t rd = fread(buf, sizeof(float), (size_t)expected_size, f);
    fclose(f);
    if ((int)rd != expected_size) {
        fprintf(stderr, "[zorn_layers] warning: '%s' has %zu floats, expected %d\n", path, rd, expected_size);
    }
    return buf;
}

int zorn_layers_load(ZORN_Engine *engine, const char *model_dir) {
    char manifest_path[ZORN_MAX_PATH];
    snprintf(manifest_path, sizeof(manifest_path), "%s/zorn_manifest.json", model_dir);

    char *manifest = zorn_read_whole_file_text(manifest_path);
    if (!manifest) {
        fprintf(stderr, "[zorn_layers] cannot read manifest '%s'\n", manifest_path);
        return -1;
    }

    
    char *config_block = zorn_json_find_block(manifest, "config");
    if (config_block) {
        size_t clen = zorn_json_block_len(config_block);
        char *cbuf = malloc(clen + 1);
        memcpy(cbuf, config_block, clen);
        cbuf[clen] = '\0';

        engine->config.hidden_size = (int)zorn_json_find_number(cbuf, "hidden_size", engine->config.hidden_size);
        engine->config.num_layers = (int)zorn_json_find_number(cbuf, "num_layers", engine->config.num_layers);
        engine->config.vocab_size = (int)zorn_json_find_number(cbuf, "vocab_size", engine->config.vocab_size);
        engine->config.max_context_length = (int)zorn_json_find_number(cbuf, "max_context_length", engine->config.max_context_length);
        engine->config.num_attn_heads = (int)zorn_json_find_number(cbuf, "num_attn_heads", engine->config.num_attn_heads);
        engine->config.num_kv_heads = (int)zorn_json_find_number(cbuf, "num_kv_heads", engine->config.num_kv_heads);
        engine->config.head_dim = (int)zorn_json_find_number(cbuf, "head_dim", engine->config.head_dim);
        engine->config.mlp_intermediate = (int)zorn_json_find_number(cbuf, "mlp_intermediate", engine->config.mlp_intermediate);
        engine->config.rope_theta = zorn_json_find_number(cbuf, "rope_theta", engine->config.rope_theta);
        engine->config.rmsnorm_eps = zorn_json_find_number(cbuf, "rmsnorm_eps", engine->config.rmsnorm_eps);
        engine->config.q_dim = engine->config.num_attn_heads * engine->config.head_dim;
        engine->config.kv_dim = engine->config.num_kv_heads * engine->config.head_dim;

        free(cbuf);
    }

    int num_layers = engine->config.num_layers;
    int hidden_size = engine->config.hidden_size;

    
    char weights_path[ZORN_MAX_PATH];
    snprintf(weights_path, sizeof(weights_path), "%s/weights.zrn", model_dir);
    size_t weights_size = 0;
    int weights_is_mmap = 0;
    void *map = zorn_mmap_readonly(weights_path, &weights_size, &weights_is_mmap);
    if (!map) {
        fprintf(stderr, "[zorn_layers] cannot read '%s'\n", weights_path);
        free(manifest);
        return -1;
    }
    engine->shared_weights_mmap_base = map;
    engine->shared_weights_mmap_size = weights_size;
    engine->shared_weights_is_mmap = weights_is_mmap;
    fprintf(stderr, "[zorn_layers] weights.zrn %s (%.1f MB)\n",
            weights_is_mmap ? "memory-mapped" : "loaded to heap (mmap unavailable)",
            (double)weights_size / (1024.0 * 1024.0));

    char scales_path[ZORN_MAX_PATH];
    snprintf(scales_path, sizeof(scales_path), "%s/scales.zrn", model_dir);
    FILE *scale_test = fopen(scales_path, "rb");
    if (!scale_test) scales_path[0] = '\0';
    else fclose(scale_test);

    engine->layers = calloc((size_t)num_layers, sizeof(ZORN_TransformerLayer));
    if (!engine->layers && num_layers > 0) {
        zorn_unmap_or_free(map, weights_size, weights_is_mmap);
        engine->shared_weights_mmap_base = NULL;
        engine->shared_weights_mmap_size = 0;
        free(manifest);
        return -1;
    }

    char *layers_array_start = zorn_json_find_block(manifest, "layers");
    
    char *layers_key = strstr(manifest, "\"layers\"");
    char *arr_start = layers_key ? strchr(layers_key, '[') : NULL;
    (void)layers_array_start;

    if (!arr_start) {
        fprintf(stderr, "[zorn_layers] no 'layers' array in manifest\n");
        zorn_unmap_or_free(map, weights_size, weights_is_mmap);
        engine->shared_weights_mmap_base = NULL;
        engine->shared_weights_mmap_size = 0;
        free(manifest);
        return -1;
    }

    char *cursor = arr_start + 1;
    int loaded_count = 0;

    for (int li = 0; li < num_layers; li++) {
        char *obj_start = strchr(cursor, '{');
        if (!obj_start) break;
        size_t obj_len = zorn_json_block_len(obj_start);

        char *obj_buf = malloc(obj_len + 1);
        memcpy(obj_buf, obj_start, obj_len);
        obj_buf[obj_len] = '\0';

        ZORN_TransformerLayer *L = &engine->layers[li];

        char ln_file[256], post_ln_file[256];
        int has_ln = (zorn_json_find_string(obj_buf, "input_layernorm", ln_file, sizeof(ln_file)) == 0);
        int has_post_ln = (zorn_json_find_string(obj_buf, "post_attention_layernorm", post_ln_file, sizeof(post_ln_file)) == 0);

        L->input_layernorm_weight = has_ln ? zorn_load_fp32_vector(model_dir, ln_file, hidden_size) : NULL;
        L->post_attention_layernorm_weight = has_post_ln ? zorn_load_fp32_vector(model_dir, post_ln_file, hidden_size) : NULL;

        int ok = 1;
        ok &= (zorn_parse_ternary_field(obj_buf, obj_len, "q_proj", &L->q_proj, map, weights_size, scales_path) == 0);
        ok &= (zorn_parse_ternary_field(obj_buf, obj_len, "k_proj", &L->k_proj, map, weights_size, scales_path) == 0);
        ok &= (zorn_parse_ternary_field(obj_buf, obj_len, "v_proj", &L->v_proj, map, weights_size, scales_path) == 0);
        ok &= (zorn_parse_ternary_field(obj_buf, obj_len, "o_proj", &L->o_proj, map, weights_size, scales_path) == 0);
        ok &= (zorn_parse_ternary_field(obj_buf, obj_len, "gate_proj", &L->gate_proj, map, weights_size, scales_path) == 0);
        ok &= (zorn_parse_ternary_field(obj_buf, obj_len, "up_proj", &L->up_proj, map, weights_size, scales_path) == 0);
        ok &= (zorn_parse_ternary_field(obj_buf, obj_len, "down_proj", &L->down_proj, map, weights_size, scales_path) == 0);

        if (ok && L->input_layernorm_weight && L->post_attention_layernorm_weight) loaded_count++;
        else fprintf(stderr, "[zorn_layers] layer %d incomplete/failed to load\n", li);

        free(obj_buf);
        cursor = obj_start + obj_len;
    }

    free(manifest);
    
    engine->layers_loaded = loaded_count;

    return (loaded_count == num_layers) ? 0 : -1;
}

void zorn_layers_free(ZORN_Engine *engine) {
    if (!engine || !engine->layers) return;
    for (int i = 0; i < engine->config.num_layers; i++) {
        free(engine->layers[i].input_layernorm_weight);
        free(engine->layers[i].post_attention_layernorm_weight);
        free(engine->layers[i].q_proj.scales);
        free(engine->layers[i].k_proj.scales);
        free(engine->layers[i].v_proj.scales);
        free(engine->layers[i].o_proj.scales);
        free(engine->layers[i].gate_proj.scales);
        free(engine->layers[i].up_proj.scales);
        free(engine->layers[i].down_proj.scales);
        
    }
    free(engine->layers);
    engine->layers = NULL;
}

int zorn_embedding_load(ZORN_Embedding *emb, const char *path, size_t vocab_size, size_t hidden_size) {
    memset(emb, 0, sizeof(*emb));
    emb->fd = -1;
    size_t size = 0;
    int is_mmap = 0;
    void *data = zorn_mmap_readonly(path, &size, &is_mmap);
    size_t expected = vocab_size * hidden_size * sizeof(float);
    if (!data) {
        fprintf(stderr, "[zorn_embedding] cannot read '%s'\n", path);
        return -1;
    }
    if (size < expected) {
        fprintf(stderr, "[zorn_embedding] '%s' too small: %zu bytes, expected %zu\n",
                path, size, expected);
        zorn_unmap_or_free(data, size, is_mmap);
        return -1;
    }
    emb->weight = (float *)data;
    emb->mmap_base = data;
    emb->mmap_size = size;
    emb->is_mmap = is_mmap;
    emb->vocab_size = vocab_size;
    emb->hidden_size = hidden_size;
    emb->loaded = 1;
    return 0;
}

void zorn_embedding_free(ZORN_Embedding *emb) {
    if (!emb) return;
    zorn_unmap_or_free(emb->mmap_base, emb->mmap_size, emb->is_mmap);
    memset(emb, 0, sizeof(*emb));
}

int zorn_kv_cache_init(ZORN_Engine *engine) {
    int num_layers = engine->config.num_layers;
    int kv_dim = engine->config.kv_dim;
    int max_ctx = engine->config.max_context_length;

    engine->kv_cache = calloc((size_t)num_layers, sizeof(ZORN_KVCacheLayer));
    for (int i = 0; i < num_layers; i++) {
        int initial_cap = max_ctx < 256 ? max_ctx : 256;
        if (initial_cap < 1) initial_cap = 1;
        engine->kv_cache[i].k_cache = calloc((size_t)initial_cap * (size_t)kv_dim, sizeof(float));
        engine->kv_cache[i].v_cache = calloc((size_t)initial_cap * (size_t)kv_dim, sizeof(float));
        engine->kv_cache[i].capacity = initial_cap;
        engine->kv_cache[i].current_len = 0;
        if (!engine->kv_cache[i].k_cache || !engine->kv_cache[i].v_cache) {
            for (int j = 0; j <= i; ++j) {
                free(engine->kv_cache[j].k_cache);
                free(engine->kv_cache[j].v_cache);
            }
            free(engine->kv_cache);
            engine->kv_cache = NULL;
            return -1;
        }
    }
    return 0;
}

void zorn_kv_cache_free(ZORN_Engine *engine) {
    if (!engine || !engine->kv_cache) return;
    for (int i = 0; i < engine->config.num_layers; i++) {
        free(engine->kv_cache[i].k_cache);
        free(engine->kv_cache[i].v_cache);
    }
    free(engine->kv_cache);
    engine->kv_cache = NULL;
}

ZORN_ModelConfig zorn_default_config(void) {
    ZORN_ModelConfig c;
    c.hidden_size = 3072;
    c.num_layers = 40;
    c.vocab_size = 131072;
    c.max_context_length = 32768;
    c.num_attn_heads = 12;
    c.num_kv_heads = 4;
    c.head_dim = 256;
    c.q_dim = 3072;
    c.kv_dim = 1024;
    c.mlp_intermediate = 23040;
    c.rope_theta = 1000042.0;
    c.rmsnorm_eps = 1e-6;
    return c;
}
