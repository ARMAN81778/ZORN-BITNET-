cmake_minimum_required(VERSION 3.18)
project(zorn_engine C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(ZORN_BUILD_PYTHON "Build Python extension source" ON)

set(ZORN_SOURCES
    zorn/zorn_core.c zorn/zorn_weights.c
    zorn/zorn_bitnet.c zorn/zorn_rag.c zorn/zorn_tokenizer.c
    zorn/zorn_simd.c zorn/zorn_rmsnorm.c zorn/zorn_rope.c
    zorn/zorn_attention.c zorn/zorn_mlp.c zorn/zorn_layers.c
    zorn/zorn_transformer.c
)

if(ZORN_BUILD_PYTHON AND NOT ANDROID)
    find_package(Python3 COMPONENTS Development REQUIRED)
    list(APPEND ZORN_SOURCES zorn/zorn_python.c)
endif()

add_library(zorn_engine SHARED ${ZORN_SOURCES})

target_include_directories(zorn_engine PRIVATE zorn)
if(ZORN_BUILD_PYTHON AND NOT ANDROID)
    target_include_directories(zorn_engine PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(zorn_engine PRIVATE ${Python3_LIBRARIES})
    if(WIN32)
        set(ZORN_PY_SUFFIX ".pyd")
    else()
        set(ZORN_PY_SUFFIX ".so")
    endif()
    set_target_properties(zorn_engine PROPERTIES
        PREFIX ""
        SUFFIX "${ZORN_PY_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/zorn"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/zorn"
    )
endif()
if(MSVC)
    target_compile_options(zorn_engine PRIVATE /O2)
else()
    target_compile_options(zorn_engine PRIVATE -O3 -Wall)
    find_package(Threads REQUIRED)
    target_link_libraries(zorn_engine PRIVATE Threads::Threads m)
endif()

if(NOT ANDROID AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    target_compile_definitions(zorn_engine PRIVATE ZORN_X86_BUILD=1)
endif()
