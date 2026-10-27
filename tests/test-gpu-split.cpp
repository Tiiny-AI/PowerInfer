#include "llama-gpu-split.h"

#include <cstdio>
#include <cstring>

struct fixture {
    ggml_context * ctx;
    ggml_tensor * index;
    ggml_tensor * bucket;

    fixture() : ctx(ggml_init({ 1024 * 1024, nullptr, false })) {
        index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        bucket = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
        ggml_set_zero(index);
        ggml_set_i32_1d(index, 0, 1);
        ggml_set_i32_1d(index, 3, 1);
        ggml_set_i32_1d(bucket, 0, 0);
        ggml_set_i32_1d(bucket, 1, 3);
    }

    ~fixture() { ggml_free(ctx); }
};

int main() {
    const char * cases[] = {
        "valid", "unsorted", "empty", "full", "negative", "upper-bound", "int-max",
        "duplicate", "unselected", "count-mismatch", "non-binary", "index-type", "bucket-type",
        "index-shape", "bucket-shape", "index-size", "oversized-bucket", "missing-index-data",
        "missing-bucket-data", "null-index", "null-bucket", "zero-source-rows", "strided-bucket"
    };
    int failures = 0;
    for (const char * name : cases) {
        fixture f;
        ggml_tensor * index = f.index;
        ggml_tensor * bucket = f.bucket;
        int64_t rows = 4;
        int64_t expected_count = 2;
        const bool expected_valid = std::strcmp(name, "valid") == 0 || std::strcmp(name, "unsorted") == 0 ||
            std::strcmp(name, "empty") == 0 || std::strcmp(name, "full") == 0;
        if (std::strcmp(name, "unsorted") == 0) {
            ggml_set_i32_1d(f.bucket, 0, 3);
            ggml_set_i32_1d(f.bucket, 1, 0);
        } else if (std::strcmp(name, "empty") == 0) {
            ggml_set_zero(f.index);
            bucket = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I32, 0);
            bucket->data = nullptr; // no dereference is needed for an empty split
            expected_count = 0;
        } else if (std::strcmp(name, "full") == 0) {
            bucket = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I32, 4);
            for (int i = 0; i < 4; ++i) {
                ggml_set_i32_1d(f.index, i, 1);
                ggml_set_i32_1d(bucket, i, i);
            }
            expected_count = 4;
        } else if (std::strcmp(name, "negative") == 0) {
            ggml_set_i32_1d(f.bucket, 1, -1);
        } else if (std::strcmp(name, "upper-bound") == 0) {
            ggml_set_i32_1d(f.bucket, 1, 4);
        } else if (std::strcmp(name, "int-max") == 0) {
            ggml_set_i32_1d(f.bucket, 1, INT32_MAX);
        } else if (std::strcmp(name, "duplicate") == 0) {
            ggml_set_i32_1d(f.bucket, 1, 0);
        } else if (std::strcmp(name, "unselected") == 0) {
            ggml_set_i32_1d(f.bucket, 1, 1);
        } else if (std::strcmp(name, "count-mismatch") == 0) {
            ggml_set_i32_1d(f.index, 1, 1);
        } else if (std::strcmp(name, "non-binary") == 0) {
            ggml_set_i32_1d(f.index, 0, 2);
        } else if (std::strcmp(name, "index-type") == 0) {
            index->type = GGML_TYPE_F32;
        } else if (std::strcmp(name, "bucket-type") == 0) {
            bucket->type = GGML_TYPE_F32;
        } else if (std::strcmp(name, "index-shape") == 0) {
            index = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, 4, 2);
        } else if (std::strcmp(name, "bucket-shape") == 0) {
            bucket = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, 2, 2);
        } else if (std::strcmp(name, "index-size") == 0) {
            rows = 3;
        } else if (std::strcmp(name, "oversized-bucket") == 0) {
            bucket = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I32, 5);
        } else if (std::strcmp(name, "missing-index-data") == 0) {
            index->data = nullptr;
        } else if (std::strcmp(name, "missing-bucket-data") == 0) {
            bucket->data = nullptr;
        } else if (std::strcmp(name, "null-index") == 0) {
            index = nullptr;
        } else if (std::strcmp(name, "null-bucket") == 0) {
            bucket = nullptr;
        } else if (std::strcmp(name, "zero-source-rows") == 0) {
            rows = 0;
        } else if (std::strcmp(name, "strided-bucket") == 0) {
            bucket->nb[0] *= 2;
        }

        int64_t selected = -1;
        std::string error;
        const bool valid = llama_gpu_split::validate(index, bucket, rows, selected, error);
        if (valid != expected_valid || (valid && selected != expected_count) ||
            (!valid && (error.empty() || selected != -1))) {
            std::fprintf(stderr, "FAIL %s: valid=%d selected=%lld error=%s\n",
                name, int(valid), static_cast<long long>(selected), error.c_str());
            ++failures;
        } else {
            std::printf("PASS %s\n", name);
        }
    }
    return failures == 0 ? 0 : 1;
}
