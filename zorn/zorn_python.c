
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "zorn.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static PyObject *py_start_chat(PyObject *self, PyObject *args, PyObject *kwargs) {
    const char *model_dir;
    int cpu_cores;
    const char *knowledge_folder = NULL;
    const char *log_path = "zorn_chat_log.jsonl";

    static char *kwlist[] = {"model_dir", "cpu_cores", "knowledge_folder", "log_path", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "si|zs", kwlist,
                                      &model_dir, &cpu_cores, &knowledge_folder, &log_path)) {
        return NULL;
    }

    ZORN_Engine engine;
    print_zorn_banner();

    if (zorn_engine_init(&engine, model_dir, cpu_cores) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "zorn_engine_init failed (see stderr)");
        return NULL;
    }

    if (knowledge_folder != NULL) {
        if (zorn_kb_load_folder(&engine.kb, knowledge_folder) != 0) {
            fprintf(stderr, "[zorn_python] warning: could not load knowledge folder '%s'\n", knowledge_folder);
        }
    }

    if (zorn_log_open(&engine, log_path) != 0) {
        fprintf(stderr, "[zorn_python] warning: logging disabled\n");
    }

    printf("\n┌────────────────────────────────────────────────────┐\n");
    printf("│        ZORN TERNARY ENGINE ACTIVE CHAT PANEL       │\n");
    printf("├────────────────────────────────────────────────────┤\n");
    printf("│ 📦 Model Name      : %-29s │\n", engine.model_name);
    printf("│ 🧱 Layers          : %-2d / %-2d loaded                 │\n", engine.layers_loaded, engine.config.num_layers);
    printf("│ 📐 Hidden/Heads    : %d / %d q-heads / %d kv-heads       │\n",
           engine.config.hidden_size, engine.config.num_attn_heads, engine.config.num_kv_heads);
    printf("│ 📚 Vocab           : %-7d                          │\n", engine.config.vocab_size);
    printf("│ 📟 CPU Threads     : %-2d (pthread + SIMD)             │\n", engine.num_cores_to_use);
    printf("│ ⚡ SIMD Level      : %-10s                       │\n", zorn_simd_level_name((int)engine.metrics.simd_level_used));
    printf("│ 📉 RAM (mmap)      : %-7.2f MB                     │\n", engine.metrics.ram_usage_mb);
    printf("│ RAG chunks      : %-7zu from folder              │\n", engine.kb.count);
    printf("│ 🧾 Log file        : %-29s │\n", engine.log_path);
    printf("│ ✅ Full generation : %-10s                       │\n", engine.real_generation_ready ? "ENABLED" : "DISABLED");
    printf("└────────────────────────────────────────────────────┘\n");
    printf("\n💬 [ZORN System] Interactive Chat Ready! Type 'exit' to close session.\n");

    char chat_buffer[ZORN_LOG_LINE_MAX];
    while (1) {
        printf("\n👤 \033[1;34mUser\033[0m: ");
        fflush(stdout);
        if (!fgets(chat_buffer, sizeof(chat_buffer), stdin)) break;
        chat_buffer[strcspn(chat_buffer, "\n")] = 0;

        if (strcmp(chat_buffer, "exit") == 0) break;
        if (strlen(chat_buffer) < 2) continue;

        zorn_execute_inference(&engine, chat_buffer);
    }

    printf("\n[System] ZORN Ternary Engine session closed.\n");
    zorn_engine_free(&engine);
    Py_RETURN_NONE;
}



typedef struct {
    ZORN_Engine engine;
} ZORN_PySession;

static void zorn_py_session_destructor(PyObject *capsule) {
    ZORN_PySession *s = (ZORN_PySession *)PyCapsule_GetPointer(capsule, "zorn.session");
    if (s) {
        zorn_engine_free(&s->engine);
        free(s);
    }
}

static PyObject *py_load_model(PyObject *self, PyObject *args, PyObject *kwargs) {
    const char *model_dir;
    int cpu_cores = 1;
    static char *kwlist[] = {"model_dir", "cpu_cores", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|i", kwlist, &model_dir, &cpu_cores)) return NULL;

    ZORN_PySession *s = calloc(1, sizeof(*s));
    if (!s) return PyErr_NoMemory();
    if (zorn_engine_init(&s->engine, model_dir, cpu_cores) != 0 || !s->engine.real_generation_ready) {
        zorn_engine_free(&s->engine);
        free(s);
        PyErr_SetString(PyExc_RuntimeError, "ZORN model initialization failed or full generation is not ready (see stderr)");
        return NULL;
    }
    return PyCapsule_New(s, "zorn.session", zorn_py_session_destructor);
}

static ZORN_PySession *zorn_get_session(PyObject *capsule) {
    return (ZORN_PySession *)PyCapsule_GetPointer(capsule, "zorn.session");
}

static void zorn_reset_cache(ZORN_Engine *e) {
    if (!e || !e->kv_cache) return;
    for (int i = 0; i < e->config.num_layers; ++i) {
        e->kv_cache[i].current_len = 0;
        size_t n = (size_t)e->kv_cache[i].capacity * (size_t)e->config.kv_dim;
        memset(e->kv_cache[i].k_cache, 0, n * sizeof(float));
        memset(e->kv_cache[i].v_cache, 0, n * sizeof(float));
    }
}


typedef struct {
    size_t id;
    float logit;
} ZORN_LogitItem;

static int zorn_logit_cmp_desc(const void *a, const void *b) {
    const ZORN_LogitItem *x = (const ZORN_LogitItem *)a;
    const ZORN_LogitItem *y = (const ZORN_LogitItem *)b;
    if (x->logit > y->logit) return -1;
    if (x->logit < y->logit) return 1;
    return 0;
}

static int zorn_contains_token(PyObject *obj, size_t token) {
    if (!obj || obj == Py_None) return 0;
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) return 0;
    Py_ssize_t n = PySequence_Size(obj);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *v = PySequence_GetItem(obj, i);
        if (!v) { PyErr_Clear(); continue; }
        long long id = PyLong_AsLongLong(v);
        Py_DECREF(v);
        if (!PyErr_Occurred() && id >= 0 && (size_t)id == token) return 1;
        PyErr_Clear();
    }
    return 0;
}

static size_t zorn_sample_advanced(const float *logits, size_t vocab_size,
                                   double temperature, int top_k, double top_p,
                                   double repetition_penalty, PyObject *history,
                                   unsigned int *rng_state) {
    if (vocab_size == 0) return 0;
    float *work = malloc(vocab_size * sizeof(float));
    ZORN_LogitItem *items = malloc(vocab_size * sizeof(ZORN_LogitItem));
    if (!work || !items) { free(work); free(items); return 0; }
    memcpy(work, logits, vocab_size * sizeof(float));

    if (repetition_penalty > 1.0 && history && (PyList_Check(history) || PyTuple_Check(history))) {
        Py_ssize_t hn = PySequence_Size(history);
        for (Py_ssize_t i = 0; i < hn; ++i) {
            PyObject *v = PySequence_GetItem(history, i);
            if (!v) { PyErr_Clear(); continue; }
            long long id = PyLong_AsLongLong(v);
            Py_DECREF(v);
            if (PyErr_Occurred()) { PyErr_Clear(); continue; }
            if (id >= 0 && (size_t)id < vocab_size) {
                if (work[id] < 0.0f) work[id] *= (float)repetition_penalty;
                else work[id] /= (float)repetition_penalty;
            }
        }
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        items[i].id = i;
        items[i].logit = work[i];
    }
    qsort(items, vocab_size, sizeof(ZORN_LogitItem), zorn_logit_cmp_desc);

    size_t keep = vocab_size;
    if (top_k > 0 && (size_t)top_k < keep) keep = (size_t)top_k;
    if (top_p > 0.0 && top_p < 1.0) {
        double max_logit = items[0].logit;
        double sum = 0.0;
        for (size_t i = 0; i < keep; ++i) sum += exp(((double)items[i].logit - max_logit) / (temperature > 1e-6 ? temperature : 1e-6));
        double cumulative = 0.0;
        size_t nucleus = 0;
        for (size_t i = 0; i < keep; ++i) {
            double p = exp(((double)items[i].logit - max_logit) / (temperature > 1e-6 ? temperature : 1e-6)) / sum;
            cumulative += p;
            nucleus = i + 1;
            if (cumulative >= top_p) break;
        }
        if (nucleus > 0) keep = nucleus;
    }
    if (keep == 0) keep = 1;

    double max_logit = items[0].logit;
    double temp = temperature > 1e-6 ? temperature : 1e-6;
    double sum_exp = 0.0;
    for (size_t i = 0; i < keep; ++i) sum_exp += exp(((double)items[i].logit - max_logit) / temp);
    double r = (double)zorn_rand_next(rng_state) / 4294967296.0;
    double cumulative = 0.0;
    size_t chosen = items[keep - 1].id;
    for (size_t i = 0; i < keep; ++i) {
        cumulative += exp(((double)items[i].logit - max_logit) / temp) / sum_exp;
        if (r <= cumulative) { chosen = items[i].id; break; }
    }
    free(work);
    free(items);
    return chosen;
}

static PyObject *py_generate(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *capsule;
    PyObject *ids_obj;
    int max_new_tokens = 128;
    double temperature = 0.7;
    int reset_cache = 1;
    int top_k = 0;
    double top_p = 1.0;
    double repetition_penalty = 1.0;
    unsigned int seed = 0;
    PyObject *stop_ids = Py_None;
    static char *kwlist[] = {"session", "input_ids", "max_new_tokens", "temperature", "reset_cache",
                             "top_k", "top_p", "repetition_penalty", "seed", "stop_ids", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|idiiddIO", kwlist,
                                      &capsule, &ids_obj, &max_new_tokens, &temperature, &reset_cache,
                                      &top_k, &top_p, &repetition_penalty, &seed, &stop_ids)) return NULL;

    ZORN_PySession *s = zorn_get_session(capsule);
    if (!s) return NULL;
    if (!PyList_Check(ids_obj) && !PyTuple_Check(ids_obj)) {
        PyErr_SetString(PyExc_TypeError, "input_ids must be a list or tuple of integers");
        return NULL;
    }
    Py_ssize_t n = PySequence_Size(ids_obj);
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "input_ids is empty");
        return NULL;
    }
    if (max_new_tokens < 0) max_new_tokens = 0;
    if (reset_cache) zorn_reset_cache(&s->engine);
    int position = reset_cache ? 0 : s->engine.kv_cache[0].current_len;
    if (position + n > s->engine.config.max_context_length) {
        PyErr_SetString(PyExc_ValueError, "input_ids exceed model context length");
        return NULL;
    }

    unsigned int rng_state = seed ? seed : ((unsigned int)time(NULL) ^ (unsigned int)(uintptr_t)s ^ (unsigned int)n);
    float *logits = NULL;
    PyObject *history = PySequence_List(ids_obj);
    if (!history) return NULL;

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(ids_obj, i);
        if (!item) { free(logits); Py_DECREF(history); return NULL; }
        long long id = PyLong_AsLongLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) { free(logits); Py_DECREF(history); return NULL; }
        if (id < 0 || (size_t)id >= s->engine.embedding.vocab_size) {
            PyErr_SetString(PyExc_ValueError, "input token id outside embedding vocabulary");
            free(logits); Py_DECREF(history); return NULL;
        }
        free(logits);
        logits = zorn_transformer_forward_token(&s->engine, (size_t)id, position++, s->engine.num_cores_to_use);
        if (!logits) {
            PyErr_SetString(PyExc_RuntimeError, "transformer forward failed");
            Py_DECREF(history); return NULL;
        }
    }

    PyObject *out = PyList_New(0);
    if (!out) { free(logits); Py_DECREF(history); return NULL; }
    int remaining = s->engine.config.max_context_length - position;
    if (remaining < 0) remaining = 0;
    if (max_new_tokens > remaining) max_new_tokens = remaining;
    for (int generated = 0; generated < max_new_tokens; ++generated) {
        size_t next = zorn_sample_advanced(logits, s->engine.tokenizer.vocab_size,
                                            temperature, top_k, top_p, repetition_penalty,
                                            history, &rng_state);
        PyObject *v = PyLong_FromSize_t(next);
        if (!v || PyList_Append(out, v) != 0) {
            Py_XDECREF(v); free(logits); Py_DECREF(out); Py_DECREF(history); return NULL;
        }
        Py_DECREF(v);
        PyObject *history_token = PyLong_FromSize_t(next);
        if (!history_token || PyList_Append(history, history_token) != 0) {
            Py_XDECREF(history_token); free(logits); Py_DECREF(out); Py_DECREF(history); return NULL;
        }
        Py_DECREF(history_token);
        const char *piece = zorn_tokenizer_decode_id(&s->engine.tokenizer, next);
        int stop = 0;
        if (piece && (strstr(piece, "<|endoftext|>") || strstr(piece, "<|eot_id|>") || strstr(piece, "</s>"))) stop = 1;
        if (zorn_contains_token(stop_ids, next)) stop = 1;
        if (stop) break;
        position = s->engine.kv_cache[0].current_len;
        if (position >= s->engine.config.max_context_length) break;
        free(logits);
        logits = zorn_transformer_forward_token(&s->engine, next, position, s->engine.num_cores_to_use);
        if (!logits) { Py_DECREF(out); Py_DECREF(history); return PyErr_NoMemory(); }
    }
    free(logits);
    Py_DECREF(history);
    return out;
}

static PyObject *py_step(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *capsule;
    unsigned long long token_id;
    int reset_cache = 0;
    double temperature = 0.7;
    int top_k = 0;
    double top_p = 1.0;
    double repetition_penalty = 1.0;
    unsigned int seed = 0;
    PyObject *history = Py_None;
    PyObject *stop_ids = Py_None;
    static char *kwlist[] = {"session", "token_id", "reset_cache", "temperature", "top_k", "top_p",
                             "repetition_penalty", "seed", "history_ids", "stop_ids", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OK|ididdIOO", kwlist, &capsule, &token_id, &reset_cache,
                                      &temperature, &top_k, &top_p, &repetition_penalty, &seed, &history, &stop_ids)) return NULL;
    ZORN_PySession *s = zorn_get_session(capsule);
    if (!s) return NULL;
    if (reset_cache) zorn_reset_cache(&s->engine);
    int position = s->engine.kv_cache[0].current_len;
    if (position >= s->engine.config.max_context_length) {
        PyErr_SetString(PyExc_ValueError, "context length exceeded"); return NULL;
    }
    if (token_id >= s->engine.embedding.vocab_size) {
        PyErr_SetString(PyExc_ValueError, "token id outside vocabulary"); return NULL;
    }
    float *logits = zorn_transformer_forward_token(&s->engine, (size_t)token_id, position, s->engine.num_cores_to_use);
    if (!logits) { PyErr_SetString(PyExc_RuntimeError, "transformer forward failed"); return NULL; }
    unsigned int rng_state = seed ? seed : ((unsigned int)time(NULL) ^ (unsigned int)(uintptr_t)s ^ (unsigned int)position);
    size_t next = zorn_sample_advanced(logits, s->engine.tokenizer.vocab_size, temperature, top_k, top_p,
                                       repetition_penalty, history, &rng_state);
    free(logits);
    PyObject *result = PyLong_FromSize_t(next);
    if (!result) return NULL;
    return result;
}

static PyObject *py_info(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    ZORN_PySession *s = zorn_get_session(capsule);
    if (!s) return NULL;
    PyObject *d = PyDict_New();
    if (!d) return NULL;
    PyDict_SetItemString(d, "model_name", PyUnicode_FromString(s->engine.model_name));
    PyDict_SetItemString(d, "model_path", PyUnicode_FromString(s->engine.model_path));
    PyDict_SetItemString(d, "layers", PyLong_FromLong(s->engine.config.num_layers));
    PyDict_SetItemString(d, "hidden_size", PyLong_FromLong(s->engine.config.hidden_size));
    PyDict_SetItemString(d, "vocab_size", PyLong_FromLong((long)s->engine.tokenizer.vocab_size));
    PyDict_SetItemString(d, "context_length", PyLong_FromLong(s->engine.config.max_context_length));
    PyDict_SetItemString(d, "threads", PyLong_FromLong(s->engine.num_cores_to_use));
    PyDict_SetItemString(d, "simd", PyUnicode_FromString(zorn_simd_level_name((int)s->engine.metrics.simd_level_used)));
    PyDict_SetItemString(d, "ram_mb", PyFloat_FromDouble(s->engine.metrics.ram_usage_mb));
    PyDict_SetItemString(d, "kv_length", PyLong_FromLong(s->engine.kv_cache ? s->engine.kv_cache[0].current_len : 0));
    PyDict_SetItemString(d, "real_generation_ready", PyBool_FromLong(s->engine.real_generation_ready));
    return d;
}

static PyObject *py_reset(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    ZORN_PySession *s = zorn_get_session(capsule);
    if (!s) return NULL;
    zorn_reset_cache(&s->engine);
    Py_RETURN_NONE;
}

static PyMethodDef ZornMethods[] = {
    {"start_chat", (PyCFunction)(void (*)(void))py_start_chat, METH_VARARGS | METH_KEYWORDS,
     "start_chat(model_dir, cpu_cores, knowledge_folder=None, log_path='zorn_chat_log.jsonl')"},
    {"load_model", (PyCFunction)(void (*)(void))py_load_model, METH_VARARGS | METH_KEYWORDS,
     "load_model(model_dir, cpu_cores=1) -> session"},
    {"generate", (PyCFunction)(void (*)(void))py_generate, METH_VARARGS | METH_KEYWORDS,
     "generate(session, input_ids, max_new_tokens=128, temperature=0.7, reset_cache=1) -> token ids"},
    {"reset", py_reset, METH_VARARGS, "reset(session)"},
    {"step", (PyCFunction)(void (*)(void))py_step, METH_VARARGS | METH_KEYWORDS, "step(session, token_id, reset_cache=False, temperature=0.7, top_k=0, top_p=1.0, repetition_penalty=1.0, seed=0, history_ids=None, stop_ids=None) -> token id"},
    {"info", py_info, METH_VARARGS, "info(session) -> dict"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef zorn_module = {
    PyModuleDef_HEAD_INIT,
    "zorn_engine",
    "ZORN Native Ternary (bitnet1582b4t) Inference Engine",
    -1,
    ZornMethods
};

PyMODINIT_FUNC PyInit_zorn_engine(void) {
    return PyModule_Create(&zorn_module);
}
