import platform
from setuptools import setup, Extension

machine = platform.machine().lower()
x86 = machine in ("x86_64", "amd64", "i386", "i686")

sources = [
    "zorn_python.c", "zorn_core.c", "zorn_weights.c", "zorn_bitnet.c",
    "zorn_rag.c", "zorn_tokenizer.c", "zorn_simd.c", "zorn_rmsnorm.c",
    "zorn_rope.c", "zorn_attention.c", "zorn_mlp.c", "zorn_layers.c",
    "zorn_transformer.c",
]

compile_args = []
link_args = []
if platform.system() != "Windows":
    compile_args += ["-O3", "-Wall"]
    compile_args += ["-pthread"]
    link_args += ["-pthread", "-lm"]
else:
    # The native code uses the Windows thread/file compatibility layer.
    compile_args += ["/O2"]

zorn_module = Extension(
    "zorn_engine",
    sources=sources,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="zorn_engine",
    version="4.2.1",
    description="ZORN portable native ternary inference engine",
    packages=["zorn"],
    package_dir={"zorn": "."},
    ext_modules=[zorn_module],
)
