#ifndef ZORN_H
#define ZORN_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "zorn_platform.h"
#include <time.h>


#define ZORN_MAX_MODEL_NAME   128
#define ZORN_MAX_PATH         512
#define ZORN_MAX_THREADS      64
#define ZORN_LOG_LINE_MAX     8192
#define ZORN_RAG_TOPK_DEFAULT 3


typedef struct {
    int    hidden_size;          /* 3072 */
    int    num_layers;           /* 40 */
    int    vocab_size;           /* 131072 */
    int    max_context_length;   /* 32768 */

    int    num_attn_heads;       /* 12 */
    int    num_kv_heads;         /* 4 */
    int    head_dim;             /* 256 */
    int    q_dim;                /* num_attn_heads * head_dim = 3072 */
    int    kv_dim;                /* num_kv_heads * head_dim = 1024 */

    int    mlp_intermediate;      /* 23040 */

    double rope_theta;            /* 1000042 */
    double rmsnorm_eps;           /* 1e-6 */
} ZORN_ModelConfig;


ZORN_ModelConfig zorn_default_config(void);


typedef struct {
    double ram_usage_mb;
    double tokens_per_sec;
    double time_to_first_token;
    long   total_tokens_generated;
    double last_matmul_ms;
    int    simd_level_used;   
} ZORN_Metrics;


typedef struct {
    uint8_t *packed_data;      
    size_t   packed_size;      
    size_t   num_weights;      
    int      is_mmap;          
    int      fd;                
    void    *mmap_base;        
    size_t   mmap_size;        
} ZORN_WeightBlock;


typedef struct {
    ZORN_WeightBlock block;
    int out_features;
    int in_features;
    float scale;                 /* fallback scalar scale */
    float *scales;               /* optional per-group I2_S scales */
    int group_size;              /* 0 = scalar scale; normally 64 for I2_S */
    int num_groups;              /* groups per output row */
    void *scale_mmap_base;
    size_t scale_mmap_size;
} ZORN_TernaryMatrix;


typedef struct {
    float *input_layernorm_weight;          /* [hidden_size], fp32 */
    float *post_attention_layernorm_weight; /* [hidden_size], fp32 */

    ZORN_TernaryMatrix q_proj;   /* [q_dim, hidden_size] */
    ZORN_TernaryMatrix k_proj;   /* [kv_dim, hidden_size] */
    ZORN_TernaryMatrix v_proj;   /* [kv_dim, hidden_size] */
    ZORN_TernaryMatrix o_proj;   /* [hidden_size, q_dim] */

    ZORN_TernaryMatrix gate_proj; /* [mlp_intermediate, hidden_size] */
    ZORN_TernaryMatrix up_proj;   /* [mlp_intermediate, hidden_size] */
    ZORN_TernaryMatrix down_proj; /* [hidden_size, mlp_intermediate] */
} ZORN_TransformerLayer;


typedef struct {
    float *k_cache;   /* [max_context_length, kv_dim] */
    float *v_cache;   /* [max_context_length, kv_dim] */
    int    current_len;
    int    capacity;
} ZORN_KVCacheLayer;


typedef struct {
    char   *content;
    size_t  content_len;
    char   *source_path;
    float  *tfidf_vector;
} ZORN_Document;


typedef struct {
    ZORN_Document *docs;
    size_t          count;
    size_t          capacity;

    char   **vocab;
    size_t   vocab_size;
    size_t   vocab_capacity;

    double  *idf;
} ZORN_KnowledgeBase;


typedef struct {
    char   **id_to_token;
    size_t   vocab_size;
    size_t   capacity;
} ZORN_Tokenizer;


typedef struct {
    float *weight;         
    void  *mmap_base;
    size_t mmap_size;
    int    fd;
    int    is_mmap;         /* 1 = real OS mmap (release via munmap), 0 = heap buffer (release via free) */
    size_t vocab_size;
    size_t hidden_size;
    int    loaded;
} ZORN_LMHead;


typedef struct {
    float *weight;   /* [vocab_size, hidden_size] */
    void  *mmap_base;
    size_t mmap_size;
    int    fd;
    int    is_mmap;         /* 1 = real OS mmap (release via munmap), 0 = heap buffer (release via free) */
    size_t vocab_size;
    size_t hidden_size;
    int    loaded;
} ZORN_Embedding;


typedef struct {
    char   model_name[ZORN_MAX_MODEL_NAME];
    char   model_path[ZORN_MAX_PATH];
    long long total_parameters;
    int    num_cores_to_use;

    ZORN_ModelConfig config;

    ZORN_Embedding   embedding;
    ZORN_TransformerLayer *layers;   
    float *final_norm_weight;        /* [hidden_size] */
    ZORN_LMHead lm_head;

    ZORN_KVCacheLayer *kv_cache;     

    
    void  *shared_weights_mmap_base;
    size_t shared_weights_mmap_size;
    int    shared_weights_is_mmap;  /* 1 = real OS mmap, 0 = heap buffer */
    void  *shared_scales_mmap_base;
    size_t shared_scales_mmap_size;

    ZORN_KnowledgeBase kb;
    ZORN_Metrics       metrics;
    ZORN_Tokenizer     tokenizer;

    int    real_generation_ready; 
    int    layers_loaded;         

    FILE  *log_fp;
    char   log_path[ZORN_MAX_PATH];
} ZORN_Engine;


typedef struct {
    const uint8_t *packed_weights;
    size_t start_byte;
    size_t end_byte;
    int64_t partial_sum;
} ZORN_MatmulJob;




int  zorn_engine_init(ZORN_Engine *engine, const char *model_dir, int cpu_cores);
void zorn_engine_free(ZORN_Engine *engine);


int  zorn_weights_load(ZORN_WeightBlock *wb, const char *bin_path);
void zorn_weights_free(ZORN_WeightBlock *wb);
int8_t zorn_unpack_ternary_weight(uint8_t packed_byte, int position);


int64_t zorn_ternary_matmul_parallel(const ZORN_WeightBlock *wb, int num_threads, double *out_elapsed_ms);


float zorn_i2s_scale_for(const ZORN_TernaryMatrix *W, size_t weight_index);
void zorn_ternary_matvec(const ZORN_TernaryMatrix *W, const float *x, float *y, int num_threads);
int  zorn_simd_detect_level(void); /* 0=scalar, 2=AVX2, 5=AVX-512 */
const char *zorn_simd_level_name(int level);

void zorn_rmsnorm(const float *x, const float *weight, float *out, int size, double eps);


void zorn_rope_apply(float *vec, int num_heads, int head_dim, int position, double theta);

void zorn_attention_forward(ZORN_Engine *engine, int layer_idx, const float *hidden_in,
                             float *hidden_out, int position, int num_threads);

void zorn_mlp_forward(ZORN_Engine *engine, int layer_idx, const float *hidden_in,
                       float *hidden_out, int num_threads);


float *zorn_transformer_forward_token(ZORN_Engine *engine, size_t token_id, int position, int num_threads);


int  zorn_layers_load(ZORN_Engine *engine, const char *model_dir);
void zorn_layers_free(ZORN_Engine *engine);
int  zorn_embedding_load(ZORN_Embedding *emb, const char *path, size_t vocab_size, size_t hidden_size);
void zorn_embedding_free(ZORN_Embedding *emb);
int  zorn_kv_cache_init(ZORN_Engine *engine);
void zorn_kv_cache_free(ZORN_Engine *engine);

int  zorn_kb_load_folder(ZORN_KnowledgeBase *kb, const char *folder_path);
void zorn_kb_free(ZORN_KnowledgeBase *kb);
int  zorn_kb_retrieve_topk(const ZORN_KnowledgeBase *kb, const char *query,
                            int k, ZORN_Document **out_docs, float *out_scores);


void zorn_execute_inference(ZORN_Engine *engine, const char *prompt);


int  zorn_tokenizer_load(ZORN_Tokenizer *tok, const char *tokenizer_json_path);
void zorn_tokenizer_free(ZORN_Tokenizer *tok);
const char *zorn_tokenizer_decode_id(const ZORN_Tokenizer *tok, size_t token_id);


size_t zorn_sample_from_logits(const float *logits, size_t vocab_size, double temperature,
                                unsigned int *rng_state);


int  zorn_log_open(ZORN_Engine *engine, const char *log_path);
void zorn_log_turn(ZORN_Engine *engine, const char *prompt,
                    ZORN_Document **retrieved, int retrieved_count, const char *response);
void zorn_log_close(ZORN_Engine *engine);


void print_zorn_banner(void);

#endif /* ZORN_H */
