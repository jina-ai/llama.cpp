#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include "ggml.h"

// parameters for logging
typedef struct {
    int start_patch;      // Starting patch index (default: 0)
    int num_patches;      // Number of patches to log (default: 5)
    int start_head;       // Starting head index (default: 0)
    int num_heads;        // Number of heads to log (default: 4)
    int start_dim;        // Starting dimension index (default: 0)
    int num_dims;         // Number of dimensions to log (default: 5)
} log_params_t;

// default parameter factory
log_params_t create_default_log_params();

// logging function
void log_to_file_or_console_parameterized(
    FILE* output_file,
    ggml_tensor* t,
    const log_params_t* params = nullptr);

// serialization function
bool write_tensor_lightbin(const char* filename, const ggml_tensor* t);

// random str/dir functions
std::string utils_random_string(std::size_t length = 12);
std::string utils_make_random_subdir(const std::string& base_dir, bool add_timestamp = false);