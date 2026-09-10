#ifndef ZORN_PLATFORM_H
#define ZORN_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#ifndef strdup
#define strdup _strdup
#endif
#include <windows.h>
#include <process.h>
typedef HANDLE zorn_thread_t;
typedef void *(*zorn_thread_fn)(void *);
typedef struct { zorn_thread_fn fn; void *arg; } zorn_thread_ctx;
static unsigned __stdcall zorn_thread_trampoline(void *p) {
    zorn_thread_ctx *ctx = (zorn_thread_ctx *)p;
    void *(*fn)(void *) = ctx->fn;
    void *arg = ctx->arg;
    free(ctx);
    (void)fn(arg);
    return 0;
}
static inline int zorn_thread_create(zorn_thread_t *t, zorn_thread_fn fn, void *arg) {
    zorn_thread_ctx *ctx = (zorn_thread_ctx *)malloc(sizeof(*ctx));
    if (!ctx) return -1;
    ctx->fn = fn; ctx->arg = arg;
    uintptr_t h = _beginthreadex(NULL, 0, zorn_thread_trampoline, ctx, 0, NULL);
    if (!h) { free(ctx); return -1; }
    *t = (HANDLE)h;
    return 0;
}
static inline int zorn_thread_join(zorn_thread_t t) {
    int rc = WaitForSingleObject(t, INFINITE) == WAIT_OBJECT_0 ? 0 : -1;
    CloseHandle(t);
    return rc;
}
static inline double zorn_time_seconds(void) {
    static LARGE_INTEGER freq;
    static int initialized = 0;
    LARGE_INTEGER now;
    if (!initialized) { QueryPerformanceFrequency(&freq); initialized = 1; }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}
#else
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
typedef pthread_t zorn_thread_t;
static inline int zorn_thread_create(pthread_t *t, void *(*fn)(void *), void *arg) {
    return pthread_create(t, NULL, fn, arg);
}
static inline int zorn_thread_join(pthread_t t) { return pthread_join(t, NULL); }
static inline double zorn_time_seconds(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    timespec_get(&ts, TIME_UTC);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
#endif

/*
 * Portable strdup. Plain "strdup" is POSIX/BSD, not standard C, so under a
 * strict C11 build (no _POSIX_C_SOURCE / _GNU_SOURCE) it may have no visible
 * declaration -- the compiler then implicitly assumes it returns "int",
 * silently truncating the returned pointer on 64-bit platforms and causing
 * memory corruption. Using our own name sidesteps that entirely, on every
 * platform.
 */
static inline char *zorn_cstrdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *out = (char *)malloc(len);
    if (out) memcpy(out, s, len);
    return out;
}

static inline unsigned int zorn_rand_next(unsigned int *state) {
    uint32_t x = *state ? *state : 0xA341316Cu;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static inline void *zorn_read_binary(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    size_t size = (size_t)n;
    void *buf = malloc(size ? size : 1);
    if (!buf) { fclose(f); return NULL; }
    if (size && fread(buf, 1, size, f) != size) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    if (out_size) *out_size = size;
    return buf;
}

/*
 * Real OS-level read-only memory mapping, used for large weight files
 * (weights.zrn, embedding.bin, lm_head.bin) so the whole model does not
 * have to be copied into the process heap. Falls back to a heap read
 * (malloc+fread) only if native mmap is unavailable or fails, e.g. on
 * some restricted filesystems (network mounts, some Colab/Docker overlay
 * setups). *out_is_mmap tells the caller which release path to use:
 * 1 -> must be released with zorn_unmap_or_free (munmap/UnmapViewOfFile)
 * 0 -> must be released with zorn_unmap_or_free (free)
 * Either way, callers should always release through zorn_unmap_or_free.
 */
static inline void *zorn_mmap_readonly(const char *path, size_t *out_size, int *out_is_mmap) {
    if (out_is_mmap) *out_is_mmap = 0;
    if (out_size) *out_size = 0;

#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return NULL;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file, &sz)) { CloseHandle(file); return NULL; }
    size_t size = (size_t)sz.QuadPart;

    if (size == 0) {
        CloseHandle(file);
        void *buf = malloc(1);
        if (out_size) *out_size = 0;
        if (out_is_mmap) *out_is_mmap = 0;
        return buf;
    }

    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) {
        /* fall back to a plain heap read */
        CloseHandle(file);
        return zorn_read_binary(path, out_size);
    }
    void *view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    /* the mapping stays valid after closing these handles, once mapped */
    CloseHandle(mapping);
    CloseHandle(file);
    if (!view) return zorn_read_binary(path, out_size);

    if (out_size) *out_size = size;
    if (out_is_mmap) *out_is_mmap = 1;
    return view;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    size_t size = (size_t)st.st_size;

    if (size == 0) {
        close(fd);
        void *buf = malloc(1);
        if (out_size) *out_size = 0;
        if (out_is_mmap) *out_is_mmap = 0;
        return buf;
    }

    void *p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* the mapping remains valid after fd is closed */
    if (p == MAP_FAILED) {
        /* fall back to a plain heap read (e.g. mmap unsupported on this fs) */
        return zorn_read_binary(path, out_size);
    }

#if defined(MADV_WILLNEED)
    madvise(p, size, MADV_WILLNEED);
#endif

    if (out_size) *out_size = size;
    if (out_is_mmap) *out_is_mmap = 1;
    return p;
#endif
}

static inline void zorn_unmap_or_free(void *ptr, size_t size, int is_mmap) {
    if (!ptr) return;
    if (is_mmap) {
#ifdef _WIN32
        UnmapViewOfFile(ptr);
#else
        munmap(ptr, size ? size : 1);
#endif
    } else {
        free(ptr);
    }
}

#endif
