# ZORN cross-platform smoke tests

## Linux/macOS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
PYTHONPATH=. python -c "import zorn; import zorn.zorn_engine; print('ZORN import OK')"

## Windows
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

## Android ARM64
cmake -S . -B build/android-arm64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DZORN_BUILD_PYTHON=OFF
cmake --build build/android-arm64

## Raw BitNet/Safetensors
python zorn/convert_safetensors_to_zrn.py MODEL_DIR OUTPUT_DIR

## GGUF
pip install -r requirements-gguf.txt
python -c "from zorn import ZornChat; ZornChat('model.gguf')"
