
#include "zorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zorn_platform.h"
#include <errno.h>

#define ZORN_MAGIC "ZRN1"
#define ZORN_HDR_SIZE 16

int8_t zorn_unpack_ternary_weight(uint8_t packed_byte, int position) {
    uint8_t bits = (packed_byte >> (position * 2)) & 0x03;
    if (bits == 1) return 1;
    if (bits == 2) return -1;
    return 0;
}


static void *zorn_matmul_worker(void *arg) {
    ZORN_MatmulJob *job = (ZORN_MatmulJob *)arg;
    int64_t acc = 0;
    for (size_t i = job->start_byte; i < job->end_byte; i++) {
        uint8_t byte = job->packed_weights[i];
        int8_t w0 = zorn_unpack_ternary_weight(byte, 0);
        int8_t w1 = zorn_unpack_ternary_weight(byte, 1);
        int8_t w2 = zorn_unpack_ternary_weight(byte, 2);
        int8_t w3 = zorn_unpack_ternary_weight(byte, 3);
        acc += (int64_t)w0 + w1 + w2 + w3;
    }
    job->partial_sum = acc;
    return NULL;
}

int64_t zorn_ternary_matmul_parallel(const ZORN_WeightBlock *wb, int num_threads, double *out_elapsed_ms) {
    if (!wb || !wb->packed_data || wb->packed_size == 0) return 0;

    if (num_threads < 1) num_threads = 1;
    if (num_threads > ZORN_MAX_THREADS) num_threads = ZORN_MAX_THREADS;
    if ((size_t)num_threads > wb->packed_size) num_threads = (int)wb->packed_size;

    zorn_thread_t threads[ZORN_MAX_THREADS];
    ZORN_MatmulJob jobs[ZORN_MAX_THREADS];

    size_t chunk = wb->packed_size / num_threads;
    size_t remainder = wb->packed_size % num_threads;

    double t0 = zorn_time_seconds();

    size_t cursor = 0;
    for (int i = 0; i < num_threads; i++) {
        size_t this_chunk = chunk + (i < (int)remainder ? 1 : 0);
        jobs[i].packed_weights = wb->packed_data;
        jobs[i].start_byte = cursor;
        jobs[i].end_byte = cursor + this_chunk;
        jobs[i].partial_sum = 0;
        cursor += this_chunk;
        zorn_thread_create(&threads[i], zorn_matmul_worker, &jobs[i]);
    }

    int64_t total = 0;
    for (int i = 0; i < num_threads; i++) {
        zorn_thread_join(threads[i]);
        total += jobs[i].partial_sum;
    }

    double t1 = zorn_time_seconds();
    double ms = (t1 - t0) * 1000.0;
    if (out_elapsed_ms) *out_elapsed_ms = ms;

    return total;
}
