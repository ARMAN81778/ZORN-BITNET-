
#include "zorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#define ZORN_CHUNK_SIZE 500
#define ZORN_CHUNK_OVERLAP 50
#define ZORN_INITIAL_CAPACITY 16



static char *zorn_strdup(const char *s, size_t n) {
    char *out = malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}



static int zorn_kb_ensure_capacity(ZORN_KnowledgeBase *kb) {
    if (kb->count < kb->capacity) return 0;
    size_t new_cap = kb->capacity == 0 ? ZORN_INITIAL_CAPACITY : kb->capacity * 2;
    ZORN_Document *tmp = realloc(kb->docs, new_cap * sizeof(ZORN_Document));
    if (!tmp) return -1;
    kb->docs = tmp;
    kb->capacity = new_cap;
    return 0;
}

static int zorn_vocab_ensure_capacity(ZORN_KnowledgeBase *kb) {
    if (kb->vocab_size < kb->vocab_capacity) return 0;
    size_t new_cap = kb->vocab_capacity == 0 ? 256 : kb->vocab_capacity * 2;
    char **tmp = realloc(kb->vocab, new_cap * sizeof(char *));
    if (!tmp) return -1;
    kb->vocab = tmp;
    kb->vocab_capacity = new_cap;
    return 0;
}


static size_t zorn_vocab_lookup_or_add(ZORN_KnowledgeBase *kb, const char *word) {
    for (size_t i = 0; i < kb->vocab_size; i++) {
        if (strcmp(kb->vocab[i], word) == 0) return i;
    }
    if (zorn_vocab_ensure_capacity(kb) != 0) return (size_t)-1;
    kb->vocab[kb->vocab_size] = zorn_cstrdup(word);
    return kb->vocab_size++;
}


typedef struct { char **tokens; size_t count; } ZORN_TokenList;

static int zorn_is_word_byte(unsigned char c) {
    return isalnum(c) || (c & 0x80); 
}

static ZORN_TokenList zorn_tokenize(const char *text) {
    ZORN_TokenList result = {0};
    size_t cap = 32;
    result.tokens = malloc(cap * sizeof(char *));

    size_t len = strlen(text);
    size_t i = 0;
    while (i < len) {
        while (i < len && !zorn_is_word_byte((unsigned char)text[i])) i++;
        size_t start = i;
        while (i < len && zorn_is_word_byte((unsigned char)text[i])) i++;
        if (i > start) {
            if (result.count >= cap) {
                cap *= 2;
                result.tokens = realloc(result.tokens, cap * sizeof(char *));
            }
            char *tok = zorn_strdup(text + start, i - start);
            
            for (char *p = tok; *p; p++) {
                if ((unsigned char)*p < 0x80) *p = (char)tolower((unsigned char)*p);
            }
            result.tokens[result.count++] = tok;
        }
    }
    return result;
}

static void zorn_tokenlist_free(ZORN_TokenList *tl) {
    for (size_t i = 0; i < tl->count; i++) free(tl->tokens[i]);
    free(tl->tokens);
    tl->tokens = NULL;
    tl->count = 0;
}



static char *zorn_read_whole_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size < 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)size + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)size, f);
    buf[rd] = '\0';
    fclose(f);
    if (out_len) *out_len = rd;
    return buf;
}

static int zorn_kb_add_chunk(ZORN_KnowledgeBase *kb, const char *content, size_t len, const char *source_path) {
    if (zorn_kb_ensure_capacity(kb) != 0) return -1;
    ZORN_Document *doc = &kb->docs[kb->count];
    doc->content = zorn_strdup(content, len);
    doc->content_len = len;
    doc->source_path = zorn_cstrdup(source_path);
    doc->tfidf_vector = NULL; 
    kb->count++;
    return 0;
}

static void zorn_kb_chunk_file(ZORN_KnowledgeBase *kb, const char *filepath) {
    size_t len = 0;
    char *content = zorn_read_whole_file(filepath, &len);
    if (!content) {
        fprintf(stderr, "[zorn_rag] skip unreadable file: %s\n", filepath);
        return;
    }
    if (len == 0) { free(content); return; }

    size_t pos = 0;
    while (pos < len) {
        size_t chunk_len = (len - pos < ZORN_CHUNK_SIZE) ? (len - pos) : ZORN_CHUNK_SIZE;
        zorn_kb_add_chunk(kb, content + pos, chunk_len, filepath);
        if (pos + chunk_len >= len) break;
        pos += (ZORN_CHUNK_SIZE - ZORN_CHUNK_OVERLAP);
    }
    free(content);
}


static int zorn_is_regular_file(const char *path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISREG(st.st_mode);
#endif
}

int zorn_kb_load_folder(ZORN_KnowledgeBase *kb, const char *folder_path) {
    memset(kb, 0, sizeof(*kb));

#ifdef _WIN32
    char pattern[ZORN_MAX_PATH];
    snprintf(pattern, sizeof(pattern), "%s\\*", folder_path);
    WIN32_FIND_DATAA data;
    HANDLE h = FindFirstFileA(pattern, &data);
    if (h == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[zorn_rag] cannot open folder '%s'\n", folder_path);
        return -1;
    }
    do {
        if (strcmp(data.cFileName, ".") == 0 || strcmp(data.cFileName, "..") == 0) continue;
        char fullpath[ZORN_MAX_PATH];
        snprintf(fullpath, sizeof(fullpath), "%s\\%s", folder_path, data.cFileName);
        if (!zorn_is_regular_file(fullpath)) continue;
        zorn_kb_chunk_file(kb, fullpath);
    } while (FindNextFileA(h, &data));
    FindClose(h);
#else
    DIR *dir = opendir(folder_path);
    if (!dir) {
        fprintf(stderr, "[zorn_rag] cannot open folder '%s'\n", folder_path);
        return -1;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        char fullpath[ZORN_MAX_PATH];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", folder_path, entry->d_name);
        if (!zorn_is_regular_file(fullpath)) continue;
        zorn_kb_chunk_file(kb, fullpath);
    }
    closedir(dir);
#endif

    fprintf(stderr, "[zorn_rag] loaded %zu chunks from '%s'\n", kb->count, folder_path);
    return 0;
}



static void zorn_kb_build_tfidf(ZORN_KnowledgeBase *kb) {
    if (kb->count == 0) return;

    ZORN_TokenList *doc_tokens = malloc(kb->count * sizeof(ZORN_TokenList));

    
    for (size_t d = 0; d < kb->count; d++) {
        doc_tokens[d] = zorn_tokenize(kb->docs[d].content);
        for (size_t t = 0; t < doc_tokens[d].count; t++) {
            zorn_vocab_lookup_or_add(kb, doc_tokens[d].tokens[t]);
        }
    }

    
    size_t *df = calloc(kb->vocab_size, sizeof(size_t));
    for (size_t d = 0; d < kb->count; d++) {
        uint8_t *seen = calloc(kb->vocab_size, sizeof(uint8_t));
        for (size_t t = 0; t < doc_tokens[d].count; t++) {
            size_t idx = zorn_vocab_lookup_or_add(kb, doc_tokens[d].tokens[t]);
            if (idx != (size_t)-1 && !seen[idx]) { seen[idx] = 1; df[idx]++; }
        }
        free(seen);
    }

    kb->idf = malloc(kb->vocab_size * sizeof(double));
    for (size_t i = 0; i < kb->vocab_size; i++) {
        kb->idf[i] = log(((double)kb->count + 1.0) / ((double)df[i] + 1.0)) + 1.0;
    }
    free(df);

    
    for (size_t d = 0; d < kb->count; d++) {
        float *vec = calloc(kb->vocab_size, sizeof(float));
        size_t *tf = calloc(kb->vocab_size, sizeof(size_t));

        for (size_t t = 0; t < doc_tokens[d].count; t++) {
            size_t idx = zorn_vocab_lookup_or_add(kb, doc_tokens[d].tokens[t]);
            if (idx != (size_t)-1) tf[idx]++;
        }
        double norm = 0.0;
        for (size_t i = 0; i < kb->vocab_size; i++) {
            if (tf[i] == 0) continue;
            double val = (double)tf[i] * kb->idf[i];
            vec[i] = (float)val;
            norm += val * val;
        }
        norm = sqrt(norm);
        if (norm > 1e-9) {
            for (size_t i = 0; i < kb->vocab_size; i++) vec[i] = (float)(vec[i] / norm);
        }
        kb->docs[d].tfidf_vector = vec;
        free(tf);
        zorn_tokenlist_free(&doc_tokens[d]);
    }
    free(doc_tokens);
}



static float zorn_cosine_sim(const float *a, const float *b, size_t n) {
    double dot = 0.0;
    for (size_t i = 0; i < n; i++) dot += (double)a[i] * (double)b[i];
    return (float)dot; 
}

int zorn_kb_retrieve_topk(const ZORN_KnowledgeBase *kb, const char *query,
                           int k, ZORN_Document **out_docs, float *out_scores) {
    if (kb->count == 0) return 0;

    
    if (kb->docs[0].tfidf_vector == NULL) {
        zorn_kb_build_tfidf((ZORN_KnowledgeBase *)kb);
    }

    ZORN_TokenList qtok = zorn_tokenize(query);
    float *qvec = calloc(kb->vocab_size, sizeof(float));
    size_t *tf = calloc(kb->vocab_size, sizeof(size_t));

    for (size_t t = 0; t < qtok.count; t++) {
        for (size_t i = 0; i < kb->vocab_size; i++) {
            if (strcmp(kb->vocab[i], qtok.tokens[t]) == 0) { tf[i]++; break; }
        }
    }
    double norm = 0.0;
    for (size_t i = 0; i < kb->vocab_size; i++) {
        if (tf[i] == 0) continue;
        double val = (double)tf[i] * kb->idf[i];
        qvec[i] = (float)val;
        norm += val * val;
    }
    norm = sqrt(norm);
    if (norm > 1e-9) {
        for (size_t i = 0; i < kb->vocab_size; i++) qvec[i] = (float)(qvec[i] / norm);
    }
    free(tf);
    zorn_tokenlist_free(&qtok);

    
    size_t n = kb->count;
    float *scores = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        scores[i] = zorn_cosine_sim(qvec, kb->docs[i].tfidf_vector, kb->vocab_size);
    }

    int found = 0;
    uint8_t *used = calloc(n, sizeof(uint8_t));
    for (int j = 0; j < k; j++) {
        int best = -1;
        float best_score = -1e9f;
        for (size_t i = 0; i < n; i++) {
            if (!used[i] && scores[i] > best_score) { best_score = scores[i]; best = (int)i; }
        }
        if (best < 0 || best_score <= 0.0f) break;
        used[best] = 1;
        out_docs[found] = (ZORN_Document *)&kb->docs[best];
        out_scores[found] = best_score;
        found++;
    }

    free(used);
    free(scores);
    free(qvec);
    return found;
}

void zorn_kb_free(ZORN_KnowledgeBase *kb) {
    if (!kb) return;
    for (size_t i = 0; i < kb->count; i++) {
        free(kb->docs[i].content);
        free(kb->docs[i].source_path);
        free(kb->docs[i].tfidf_vector);
    }
    free(kb->docs);
    for (size_t i = 0; i < kb->vocab_size; i++) free(kb->vocab[i]);
    free(kb->vocab);
    free(kb->idf);
    memset(kb, 0, sizeof(*kb));
}
