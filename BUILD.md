# ZORN 4.2 portable builds

Targets:
- Linux x86_64 / ARM64
- Windows x64 (MSVC or clang-cl)
- macOS Intel / Apple Silicon
- Android ARM64 through the Android NDK
- GGUF through the optional llama-cpp-python backend
- Raw Hugging Face/Safetensors checkpoints through automatic conversion

## Python

From the `zorn` directory:

```bash
python -m pip install -r ../requirements.txt
python -m pip install .
```

On x86 Linux/macOS the build enables AVX2/FMA/AVX-512 target code where supported.
On ARM (Apple Silicon/Android) the scalar path is used automatically.

## CMake

```bash
cmake -S . -B build
cmake --build build --config Release
```

Android:

```bash
cmake -S . -B build/android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24
cmake --build build/android-arm64
```

## Model inputs

`ZornSession` accepts:
1. a normal ZORN directory;
2. a `.safetensors` file or a Hugging Face directory containing Safetensors, which is converted automatically;
3. a `.gguf` file when `llama-cpp-python` is installed.

GGUF uses the llama.cpp backend; it is not silently converted to ternary weights, because arbitrary GGUF quantization is not equivalent to BitNet ternary quantization.
