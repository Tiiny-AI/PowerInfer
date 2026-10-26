# CUDA sparse operation regression

`test-cuda-sparse` exercises the production `ggml_cuda_compute_forward` dispatcher
using synthetic tensors. No model weights or Python dependencies are required.

Build with a CUDA toolkit and a supported host C++ compiler. The example targets
an RTX 5080; change `CMAKE_CUDA_ARCHITECTURES` for a different GPU.

```sh
cmake -S . -B build-sparse-on \
  -DLLAMA_CUBLAS=ON -DLLAMA_CUDA_F16=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_STANDARD=17 \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build-sparse-on --config Release --target test-cuda-sparse
ctest --test-dir build-sparse-on -C Release -R '^test-cuda-sparse-' --output-on-failure
```

Repeat in a separate build directory with `LLAMA_CUDA_F16=OFF` to cover the
default configuration. The tests are registered only for the CUDA backend,
run serially, and return the CTest skip code 77 if no usable CUDA device exists.

The eight cases cover Q4_0 and F16 weights, one and three input vectors, and
direct and mapped output rows. Each checks active, inactive, and mixed masks
against a CPU dot-product reference using the decoded stored weights. Inputs
are chosen to avoid conflating Q8_1 input quantization with indexing errors.

After warmup, each case repeats the operation 2048 times with input width 16384
and 64 stored weight rows. Retained CUDA memory may grow by at most 16 MiB.
Run on an otherwise idle GPU: `cudaMemGetInfo` can include other processes,
and the tolerance accommodates allocator granularity and device bookkeeping.
The original F16-enabled vector paths exceed this tolerance by losing a
temporary activation buffer on every call. Batch paths are controls.

For a short sanitizer check, invoke the executable directly:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 \
  build-sparse-on/bin/test-cuda-sparse q4_0 1 1 16
```

Arguments are `q4_0|f16`, input-vector count, mapped-row flag, and optionally
iteration count. Use `bin/Release/test-cuda-sparse.exe` with a Windows
multi-configuration generator. Sixteen iterations are a sanitizer smoke test;
use the default 2048 iterations for the retained-memory regression.
