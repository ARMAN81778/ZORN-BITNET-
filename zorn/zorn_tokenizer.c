
#include "zorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define ZORN_TOK_INITIAL_CAP 4096

static int zorn_tok_ensure_capacity(ZORN_Tokenizer *tok, size_t needed_id) {
    if (needed_id < tok->capacity) return 0;
    size_t new_cap = tok->capacity == 0 ? ZORN_TOK_INITIAL_CAP : tok->capacity;
    while (new_cap <= needed_id) new_cap *= 2;
    char **tmp = realloc(tok->id_to_token, new_cap * sizeof(char *));
    if (!tmp) return -1;
    for (size_t i = tok->capacity; i < new_cap; i++) tmp[i] = NULL;
    tok->id_to_token = tmp;
    tok->capacity = new_cap;
    return 0;
}


static size_t zorn_json_unescape(const char *src, size_t src_len, char *dst, size_t dst_cap) {
    size_t si = 0, di = 0;
    while (si < src_len && di + 4 < dst_cap) {
        if (src[si] == '\\' && si + 1 < src_len) {
            char c = src[si + 1];
            if (c == 'n') { dst[di++] = '\n'; si += 2; }
            else if (c == 't') { dst[di++] = '\t'; si += 2; }
            else if (c == '"') { dst[di++] = '"'; si += 2; }
            else if (c == '\\') { dst[di++] = '\\'; si += 2; }
            else if (c == 'u' && si + 5 < src_len) {
                unsigned int cp = 0;
                sscanf(src + si + 2, "%4x", &cp);
                si += 6;
                
                if (cp < 0x80) {
                    dst[di++] = (char)cp;
                } else if (cp < 0x800) {
                    dst[di++] = (char)(0xC0 | (cp >> 6));
                    dst[di++] = (char)(0x80 | (cp & 0x3F));
                } else {
                    dst[di++] = (char)(0xE0 | (cp >> 12));
                    dst[di++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    dst[di++] = (char)(0x80 | (cp & 0x3F));
                }
            } else { dst[di++] = src[si]; si++; }
        } else {
            dst[di++] = src[si]; si++;
        }
    }
    dst[di] = '\0';
    return di;
}


static void zorn_bpe_byte_decode_inplace(char *s) {
    char *read = s, *write = s;
    while (*read) {
        if ((unsigned char)read[0] == 0xC4 && (unsigned char)read[1] == 0xA0) {
            *write++ = ' ';
            read += 2;
        } else if ((unsigned char)read[0] == 0xC4 && (unsigned char)read[1] == 0x8A) {
            *write++ = '\n';
            read += 2;
        } else {
            *write++ = *read++;
        }
    }
    *write = '\0';
}

int zorn_tokenizer_load(ZORN_Tokenizer *tok, const char *tokenizer_json_path) {
    memset(tok, 0, sizeof(*tok));

    FILE *f = fopen(tokenizer_json_path, "rb");
    if (!f) {
        fprintf(stderr, "[zorn_tokenizer] cannot open '%s'\n", tokenizer_json_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)fsize + 1);
    if (!buf) { fclose(f); return -1; }
    size_t rd = fread(buf, 1, (size_t)fsize, f);
    buf[rd] = '\0';
    fclose(f);

    
    const char *vocab_key = "\"vocab\"";
    char *vocab_start = strstr(buf, vocab_key);
    if (!vocab_start) {
        fprintf(stderr, "[zorn_tokenizer] 'vocab' key not found in %s\n", tokenizer_json_path);
        free(buf);
        return -1;
    }
    char *brace = strchr(vocab_start, '{');
    if (!brace) { free(buf); return -1; }

    
    char *p = brace + 1;
    int depth = 1;
    char token_buf[1024];
    char decoded_buf[1024];

    size_t max_id_seen = 0;

    while (*p && depth > 0) {
        if (*p == '{') { depth++; p++; continue; }
        if (*p == '}') { depth--; p++; continue; }
        if (*p == '"') {
            
            char *str_start = p + 1;
            char *q = str_start;
            while (*q && !(*q == '"' && *(q - 1) != '\\')) q++;
            size_t raw_len = (size_t)(q - str_start);
            if (raw_len >= sizeof(token_buf)) raw_len = sizeof(token_buf) - 1;

            zorn_json_unescape(str_start, raw_len, decoded_buf, sizeof(decoded_buf));
            zorn_bpe_byte_decode_inplace(decoded_buf);
            strncpy(token_buf, decoded_buf, sizeof(token_buf) - 1);
            token_buf[sizeof(token_buf) - 1] = '\0';

            p = q + 1;
            
            while (*p && (*p == ':' || *p == ' ')) p++;
            long id = strtol(p, &p, 10);
            if (id < 0) continue;

            if (zorn_tok_ensure_capacity(tok, (size_t)id) != 0) continue;
            free(tok->id_to_token[id]); 
            tok->id_to_token[id] = zorn_cstrdup(token_buf);
            if ((size_t)id > max_id_seen) max_id_seen = (size_t)id;
            continue;
        }
        p++;
    }

    tok->vocab_size = max_id_seen + 1;
    free(buf);

    fprintf(stderr, "[zorn_tokenizer] loaded %zu tokens from '%s'\n", tok->vocab_size, tokenizer_json_path);
    return (tok->vocab_size > 0) ? 0 : -1;
}

const char *zorn_tokenizer_decode_id(const ZORN_Tokenizer *tok, size_t token_id) {
    if (token_id >= tok->capacity || tok->id_to_token[token_id] == NULL) {
        return "<unk>";
    }
    return tok->id_to_token[token_id];
}

void zorn_tokenizer_free(ZORN_Tokenizer *tok) {
    if (!tok) return;
    for (size_t i = 0; i < tok->capacity; i++) free(tok->id_to_token[i]);
    free(tok->id_to_token);
    memset(tok, 0, sizeof(*tok));
}

