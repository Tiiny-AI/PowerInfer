#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

// Internal validation shared by the sidecar loader and the row-copy path.
namespace llama_gpu_split {

inline bool is_i32_vector(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_I32 && tensor->ne[0] >= 0 &&
        tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}

inline bool validate_layout(const ggml_tensor * index, const ggml_tensor * bucket,
                            int64_t rows, std::string & error) {
    if (rows <= 0 || !is_i32_vector(index) || !is_i32_vector(bucket) || index->ne[0] != rows || bucket->ne[0] > rows) {
        error = "GPU split tensors must be I32 vectors matching the FFN row count";
        return false;
    }
    return true;
}

inline bool validate_bucket(const ggml_tensor * bucket, int64_t rows, std::string & error) {
    if (rows <= 0 || !is_i32_vector(bucket) || bucket->ne[0] > rows || (bucket->ne[0] > 0 && !bucket->data)) {
        error = "invalid GPU bucket shape, type, or data";
        return false;
    }
    const int32_t * data = static_cast<const int32_t *>(bucket->data);
    for (int64_t i = 0; i < bucket->ne[0]; ++i) {
        if (data[i] < 0 || int64_t(data[i]) >= rows) {
            error = "gpu_bucket[" + std::to_string(i) + "]=" + std::to_string(data[i]) +
                " is outside [0, " + std::to_string(rows) + ")";
            return false;
        }
    }
    return true;
}

inline bool validate(const ggml_tensor * index, const ggml_tensor * bucket,
                     int64_t rows, int64_t & selected, std::string & error) {
    if (!validate_layout(index, bucket, rows, error) || !validate_bucket(bucket, rows, error)) {
        return false;
    }
    if (!index->data) {
        error = "missing GPU index data";
        return false;
    }
    const int32_t * mask = static_cast<const int32_t *>(index->data);
    int64_t count = 0;
    for (int64_t row = 0; row < rows; ++row) {
        if (mask[row] != 0 && mask[row] != 1) {
            error = "GPU index mask must contain only zero or one";
            return false;
        }
        count += mask[row];
    }
    if (count != bucket->ne[0]) {
        error = "GPU bucket length does not match the number of selected neurons";
        return false;
    }
    std::vector<bool> seen(static_cast<size_t>(rows), false);
    const int32_t * data = static_cast<const int32_t *>(bucket->data);
    for (int64_t i = 0; i < count; ++i) {
        const int32_t row = data[i]; // bounds were checked above
        if (seen[row] || mask[row] != 1) {
            error = "GPU bucket contains a duplicate or an unselected row";
            return false;
        }
        seen[row] = true;
    }
    selected = count;
    return true;
}

} // namespace llama_gpu_split
