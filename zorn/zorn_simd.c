#include "zorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#define ZORN_X86 1

static unsigned long long zorn_xgetbv0(void) {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, edx;
    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
#else
    return 0;
#endif
}
#endif

int zorn_simd_detect_level(void) {
#ifdef ZORN_X86
    static int cached = -1;
    if (cached >= 0) return cached;

    int level = 0;
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, 1, 0);
    ecx = (unsigned int)regs[2];
    if ((ecx & (1u << 27)) && (ecx & (1u << 28))) {
        unsigned long long xcr0 = zorn_xgetbv0();
        if ((xcr0 & 0x6) == 0x6) {
            __cpuidex(regs, 7, 0);
            ebx = (unsigned int)regs[1];
            if (ebx & (1u << 5)) level = 2; /* AVX2 */
            if ((xcr0 & 0xE0) == 0xE0 &&
                (ebx & (1u << 16)) && (ebx & (1u << 30))) level = 5; /* AVX-512F+BW */
        }
    }
#else
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) &&
        (ecx & (1u << 27)) && (ecx & (1u << 28))) {
        unsigned long long xcr0 = zorn_xgetbv0();
        if ((xcr0 & 0x6) == 0x6 &&
            __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            if (ebx & (1u << 5)) level = 2;
            if ((xcr0 & 0xE0) == 0xE0 &&
                (ebx & (1u << 16)) && (ebx & (1u << 30))) level = 5;
        }
    }
#endif
    cached = level;
    return level;
#else
    return 0;
#endif
}
const char *zorn_simd_level_name(int level){return level>=5?"AVX-512":level>=2?"AVX2":"scalar";}

static inline int8_t decode2(uint8_t b,int pos){
    unsigned c=(b>>(2*pos))&3u;
    return c==1?1:c==2?-1:0;
}

#ifdef ZORN_X86
#if defined(_MSC_VER)
#define ZORN_TARGET_AVX2
#define ZORN_TARGET_AVX512
#else
#define ZORN_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define ZORN_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,fma")))
#endif
ZORN_TARGET_AVX2
static float dot_avx2(const int8_t*w,const float*x,int n){
    __m256 acc=_mm256_setzero_ps(); int i=0;
    for(;i+8<=n;i+=8){
        __m128i q=_mm_loadl_epi64((const __m128i*)(w+i));
        __m256i wi=_mm256_cvtepi8_epi32(q);
        acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(wi),_mm256_loadu_ps(x+i),acc);
    }
    __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
    __m128 s=_mm_add_ps(lo,hi); s=_mm_add_ps(s,_mm_movehl_ps(s,s)); s=_mm_add_ss(s,_mm_shuffle_ps(s,s,1));
    float r=_mm_cvtss_f32(s); for(;i<n;i++) r+=(float)w[i]*x[i]; return r;
}
ZORN_TARGET_AVX512
static float dot_avx512(const int8_t*w,const float*x,int n){
    __m512 acc=_mm512_setzero_ps(); int i=0;
    for(;i+16<=n;i+=16){
        __m128i q=_mm_loadu_si128((const __m128i*)(w+i));
        __m512i wi=_mm512_cvtepi8_epi32(q);
        acc=_mm512_fmadd_ps(_mm512_cvtepi32_ps(wi),_mm512_loadu_ps(x+i),acc);
    }
    float r=_mm512_reduce_add_ps(acc); for(;i<n;i++) r+=(float)w[i]*x[i]; return r;
}
#endif

static void unpack_group(const uint8_t*packed,size_t start,int n,int8_t*out){
    for(int i=0;i<n;i++){size_t idx=start+(size_t)i; out[i]=decode2(packed[idx>>2],idx&3);}
}

static void *worker(void*arg){
    typedef struct {const ZORN_TernaryMatrix*W;const float*x;float*y;int a,b,simd;} Job;
    Job*j=(Job*)arg; const ZORN_TernaryMatrix*W=j->W; int n=W->in_features;
    int8_t*buf=(int8_t*)malloc((size_t)n); if(!buf)return NULL;
    for(int r=j->a;r<j->b;r++){
        size_t row=(size_t)r*(size_t)n; unpack_group(W->block.packed_data,row,n,buf);
        float dot=0.0f;
        if(W->scales && W->group_size>0){
            for(int base=0;base<n;base+=W->group_size){
                int len=W->group_size; if(base+len>n)len=n-base;
                float d=0.0f;
#ifdef ZORN_X86
                if(j->simd>=5) d=dot_avx512(buf+base,j->x+base,len);
                else if(j->simd>=2) d=dot_avx2(buf+base,j->x+base,len);
                else
#endif
                { for(int i=0;i<len;i++) d+=(float)buf[base+i]*j->x[base+i]; }
                dot += d*zorn_i2s_scale_for(W,row+(size_t)base);
            }
        } else {
#ifdef ZORN_X86
            if(j->simd>=5) dot=dot_avx512(buf,j->x,n);
            else if(j->simd>=2) dot=dot_avx2(buf,j->x,n);
            else
#endif
            { for(int i=0;i<n;i++) dot+=(float)buf[i]*j->x[i]; }
            dot*=W->scale;
        }
        j->y[r]=dot;
    }
    free(buf); return NULL;
}

void zorn_ternary_matvec(const ZORN_TernaryMatrix*W,const float*x,float*y,int num_threads){
    if(!W||!x||!y||W->out_features<=0||W->in_features<=0)return;
    if (num_threads < 1) num_threads = 1;
    if (num_threads > ZORN_MAX_THREADS) num_threads = ZORN_MAX_THREADS;
    if(num_threads>W->out_features)num_threads=W->out_features;
    typedef struct {const ZORN_TernaryMatrix*W;const float*x;float*y;int a,b,simd;} Job;
    zorn_thread_t th[ZORN_MAX_THREADS]; Job jobs[ZORN_MAX_THREADS]; int simd=zorn_simd_detect_level();
    int q=W->out_features/num_threads, rem=W->out_features%num_threads, cur=0;
    for(int t=0;t<num_threads;t++){int nr=q+(t<rem);jobs[t]=(Job){W,x,y,cur,cur+nr,simd};zorn_thread_create(&th[t],worker,&jobs[t]);cur+=nr;}
    for(int t=0;t<num_threads;t++)zorn_thread_join(th[t]);
}
