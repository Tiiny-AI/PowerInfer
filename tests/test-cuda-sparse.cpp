#include "ggml.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Exercise the actual CUDA dispatcher without loading a model. Run on an idle GPU:
// cudaMemGetInfo includes allocations made by other processes on some platforms.
#define CHECK_CUDA(call) do { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

static void upload(ggml_tensor * tensor) {
    tensor->backend = GGML_BACKEND_GPU;
    ggml_cuda_transform_tensor(tensor->data, tensor);
}

static bool compute(ggml_tensor * output) {
    ggml_compute_params params = {};
    params.type = GGML_TASK_COMPUTE;
    params.nth = 1;
    if (!ggml_cuda_compute_forward(&params, output)) {
        std::fprintf(stderr, "Sparse operation did not run on CUDA\n");
        return false;
    }
    return true;
}

static bool check_output(ggml_tensor * output, const std::vector<float> & weights,
                         const float * input, const float * mask,
                         int k, int rows, int columns, bool mapped) {
    CHECK_CUDA(cudaDeviceSynchronize());
    ggml_cuda_copy_to_host(output);
    const float * actual = static_cast<const float *>(output->data);
    const int out_rows = static_cast<int>(output->ne[0]);
    std::vector<double> expected(out_rows * columns, 0.0);
    for (int col = 0; col < columns; ++col) {
        for (int row = 0; row < rows; ++row) {
            const int out_row = mapped ? 2 * row + 1 : row;
            if (mask[col * out_rows + out_row] < 0.5f) {
                continue;
            }
            double sum = 0.0;
            for (int i = 0; i < k; ++i) {
                sum += double(weights[row * k + i]) * input[col * k + i];
            }
            expected[col * out_rows + out_row] = sum;
        }
    }
    double max_error = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::fabs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
        // The F16 vector kernel accumulates in half2; Q4_0 vectors quantize input.
        if (!std::isfinite(actual[i]) || error > 0.01 + 0.01 * std::fabs(expected[i])) {
            std::fprintf(stderr, "Output mismatch at %zu: actual=%g expected=%.9g\n",
                         i, actual[i], expected[i]);
            return false;
        }
        if (expected[i] == 0.0 && mask[i] < 0.5f && actual[i] != 0.0f) {
            std::fprintf(stderr, "Inactive row was not cleared at %zu\n", i);
            return false;
        }
    }
    std::printf("max_abs_error=%.9g\n", max_error);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5 ||
        (std::strcmp(argv[1], "q4_0") != 0 && std::strcmp(argv[1], "f16") != 0)) {
        std::fprintf(stderr, "Usage: %s q4_0|f16 columns mapped [iterations]\n", argv[0]);
        return 1;
    }
    const ggml_type type = std::strcmp(argv[1], "q4_0") == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
    const int columns = std::atoi(argv[2]);
    const bool mapped = std::atoi(argv[3]) != 0;
    const int iterations = argc == 5 ? std::atoi(argv[4]) : 2048;
    if (columns < 1 || columns > 4 || iterations < 1 || iterations > 16384) {
        return 1;
    }
    int devices = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&devices);
    if (device_status != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "No usable CUDA device: skipping\n");
        return 77;
    }
    CHECK_CUDA(cudaSetDevice(0));
    const int k = 16384;
    const int rows = 64;
    const int out_rows = mapped ? 2 * rows : rows;
    ggml_init_params init = { 32 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(init);
    if (!ctx || !ggml_cublas_loaded()) {
        std::fprintf(stderr, "Could not initialize ggml CUDA\n");
        return 1;
    }
    ggml_cuda_set_device_constants(0.5f);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, k, rows);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, columns);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_rows, columns);
    ggml_tensor * lookup = mapped ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows) : nullptr;
    ggml_tensor * output = mapped
        ? ggml_mul_mat_idx_upscale(ctx, weight, input, mask, lookup, out_rows)
        : ggml_mul_mat_idx(ctx, weight, input, mask, nullptr);

    std::vector<float> reference_weights(k * rows);
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < k; ++i) {
            reference_weights[row * k + i] = float((i * 13 + row * 7) % 17 - 8) / 128.0f;
        }
    }
    const ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
    traits.from_float_reference(reference_weights.data(), weight->data, k * rows);
    traits.to_float(weight->data, reference_weights.data(), k * rows);
    float * x = static_cast<float *>(input->data);
    for (int col = 0; col < columns; ++col) {
        for (int i = 0; i < k; ++i) {
            // Exactly representable inputs make the reference insensitive to Q8_1 rounding.
            x[col * k + i] = ((i / 32 + col) % 2 ? -1.0f : 1.0f) * (col + 1) / 8.0f;
        }
    }
    if (mapped) {
        int32_t * indices = static_cast<int32_t *>(lookup->data);
        for (int row = 0; row < rows; ++row) {
            indices[row] = 2 * row + 1;
        }
    }
    std::memset(mask->data, 0, ggml_nbytes(mask));
    std::memset(output->data, 0, ggml_nbytes(output));
    upload(weight);
    upload(input);
    upload(mask);
    upload(output);
    if (lookup) {
        upload(lookup);
    }

    bool ok = true;
    float * mask_data = static_cast<float *>(mask->data);
    // Active -> inactive -> mixed catches stale outputs as well as indexing errors.
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (int col = 0; col < columns; ++col) {
            for (int row = 0; row < out_rows; ++row) {
                mask_data[col * out_rows + row] = pattern == 0 ? 1.0f :
                    pattern == 1 ? 0.0f : float((row + col) % 3 != 0);
            }
        }
        ggml_cuda_copy_to_device(mask);
        ok = compute(output) && ok;
        ok = check_output(output, reference_weights, x, mask_data, k, rows, columns, mapped) && ok;
    }
    // Warm kernels and reusable allocator buffers before measuring retained memory.
    for (int i = 0; i < 32; ++i) {
        ok = compute(output) && ok;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    size_t free_before, free_after, total;
    CHECK_CUDA(cudaMemGetInfo(&free_before, &total));
    for (int i = 0; i < iterations; ++i) {
        ok = compute(output) && ok;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemGetInfo(&free_after, &total));
    ok = check_output(output, reference_weights, x, mask_data, k, rows, columns, mapped) && ok;
    const size_t growth = free_before > free_after ? free_before - free_after : 0;
    const size_t allowance = 16 * 1024 * 1024;
    std::printf("type=%s columns=%d mapped=%d iterations=%d free_before=%zu free_after=%zu growth_bytes=%zu\n",
                argv[1], columns, int(mapped), iterations, free_before, free_after, growth);
    if (growth > allowance) {
        std::fprintf(stderr, "CUDA memory grew by %.2f MiB after warmup (limit 16 MiB)\n",
                     double(growth) / (1024 * 1024));
        ok = false;
    }
    for (ggml_tensor * tensor : { weight, input, mask, output, lookup }) {
        if (tensor) {
            ggml_cuda_free_data(tensor);
        }
    }
    ggml_free(ctx);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
