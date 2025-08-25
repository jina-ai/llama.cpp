// utils.cpp
#include "llama-utils.h"   // your own header

#include "ggml.h"          // project dependency headers

#include <algorithm>       // std::max, std::min
#include <cerrno>          // errno
#include <chrono>          // time utilities
#include <cstdint>         // int32_t, int64_t
#include <cstdio>          // printf, fprintf, fopen, fwrite, fclose
#include <cstring>         // strerror
#include <filesystem>      // std::filesystem
#include <iomanip>         // std::put_time
#include <random>          // RNG
#include <sstream>         // stringstream
#include <stdexcept>       // std::runtime_error

std::string utils_random_string(std::size_t length) {
    if (length == 0) length = 1;
    static const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(0, static_cast<int>(sizeof(charset) - 2));
    std::string s;
    s.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        s.push_back(charset[dist(rng)]);
    }
    return s;
}

static std::string utils_timestamp_utc_compact() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto tt  = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string utils_make_random_subdir(const std::string& base_dir, bool add_timestamp) {
    namespace fs = std::filesystem;

    if (base_dir.empty()) {
        throw std::runtime_error("utils_make_random_subdir: base_dir is empty");
    }

    std::string base = base_dir;
    if (!base.empty() && base.back() != '/' && base.back() != '\\') {
        base += '/';
    }

    std::string name;
    if (add_timestamp) {
        name += utils_timestamp_utc_compact();
        name += "_";
    }
    name += utils_random_string(12);

    const fs::path full = fs::path(base) / name;

    std::error_code ec;
    if (!fs::create_directories(full, ec) && ec) {
        std::ostringstream oss;
        oss << "utils_make_random_subdir: failed to create '" << full.string()
            << "': " << ec.message();
        throw std::runtime_error(oss.str());
    }

    return full.string();
}

log_params_t create_default_log_params() {
    log_params_t params = {0};
    params.start_patch = 0;
    params.num_patches = 5;
    params.start_head = 0;
    params.num_heads = 5;
    params.start_dim = 0;
    params.num_dims = 10;
    return params;
}

void log_to_file_or_console_parameterized(
    FILE* output_file, 
    ggml_tensor* t,
    const log_params_t* params
) {
    if (!t) return;
    
    // Use default parameters if none provided
    log_params_t default_params = create_default_log_params();
    if (!params) params = &default_params;
    
    #define PRINT_TO_OUTPUT(format, ...) \
        do { \
            if (output_file) fprintf(output_file, format, ##__VA_ARGS__); \
            else printf(format, ##__VA_ARGS__); \
        } while (0)
    
    PRINT_TO_OUTPUT("=== %s === Shape: [", t->name);
    for (int d = 0; d < GGML_MAX_DIMS && t->ne[d] > 0; d++) {
        PRINT_TO_OUTPUT("%ld", t->ne[d]);
        if (d < GGML_MAX_DIMS - 1 && t->ne[d+1] > 0) {
            PRINT_TO_OUTPUT(", ");
        }
    }
    PRINT_TO_OUTPUT("]\n");
    
    size_t tensor_size = ggml_nelements(t);
    if (tensor_size == 0) {
        PRINT_TO_OUTPUT("Empty tensor\n\n");
        return;
    }
    
    float* data = (float*)t->data;
    
    if (t->ne[2] <= 1) {
        // 2D tensor: [d_head, n_patch] in GGML
        const int d_head = t->ne[0];
        const int n_patch = t->ne[1];
        
        // Calculate patch range
        int start_patch = std::max(0, std::min(params->start_patch, n_patch - 1));
        int end_patch = std::min(start_patch + params->num_patches, n_patch);
        
        // Calculate dimension range
        int start_dim = std::max(0, std::min(params->start_dim, d_head - 1));
        int end_dim = std::min(start_dim + params->num_dims, d_head);
        
        PRINT_TO_OUTPUT("Logging patches %d-%d, dimensions %d-%d\n", 
                       start_patch, end_patch - 1, start_dim, end_dim - 1);
        
        for (int patch = start_patch; patch < end_patch; patch++) {
            PRINT_TO_OUTPUT("Patch %d: ", patch);
            
            for (int head_dim = start_dim; head_dim < end_dim; head_dim++) {
                size_t ggml_idx = patch * d_head + head_dim;
                PRINT_TO_OUTPUT("%.6f ", data[ggml_idx]);
            }
            if (end_dim < d_head) {
                PRINT_TO_OUTPUT("... (dims %d-%d)", end_dim, d_head - 1);
            }
            PRINT_TO_OUTPUT("\n");
        }
        if (end_patch < n_patch) {
            PRINT_TO_OUTPUT("... (patches %d-%d not shown)\n", end_patch, n_patch - 1);
        }
    } else {
        // 3D tensor: [d_head, n_head, n_patch] in GGML
        const int d_head = t->ne[0];
        const int n_head = t->ne[1];
        const int n_patch = t->ne[2];
        
        // Calculate ranges
        int start_patch = std::max(0, std::min(params->start_patch, n_patch - 1));
        int end_patch = std::min(start_patch + params->num_patches, n_patch);
        
        int start_head = std::max(0, std::min(params->start_head, n_head - 1));
        int end_head = std::min(start_head + params->num_heads, n_head);
        
        int start_dim = std::max(0, std::min(params->start_dim, d_head - 1));
        int end_dim = std::min(start_dim + params->num_dims, d_head);
        
        PRINT_TO_OUTPUT("Logging patches %d-%d, heads %d-%d, dimensions %d-%d\n", 
                       start_patch, end_patch - 1, start_head, end_head - 1, 
                       start_dim, end_dim - 1);
        
        for (int patch = start_patch; patch < end_patch; patch++) {
            PRINT_TO_OUTPUT("Patch %d\n", patch);
            
            for (int head = start_head; head < end_head; head++) {
                PRINT_TO_OUTPUT("  Head %d: ", head);
                
                for (int head_dim = start_dim; head_dim < end_dim; head_dim++) {
                    size_t ggml_idx = patch * (d_head * n_head) + head * d_head + head_dim;
                    PRINT_TO_OUTPUT("%.6f ", data[ggml_idx]);
                }
                if (end_dim < d_head) {
                    PRINT_TO_OUTPUT("... (dims %d-%d)", end_dim, d_head - 1);
                }
                PRINT_TO_OUTPUT("\n");
            }
            if (end_head < n_head) {
                PRINT_TO_OUTPUT("  ... (heads %d-%d not shown)\n", end_head, n_head - 1);
            }
        }
        if (end_patch < n_patch) {
            PRINT_TO_OUTPUT("... (patches %d-%d not shown)\n", end_patch, n_patch - 1);
        }
    }
    
    PRINT_TO_OUTPUT("\n");
    #undef PRINT_TO_OUTPUT
}

static inline bool is_c_contiguous(const ggml_tensor* t) {
    const size_t es = ggml_element_size(t);
    if (t->nb[0] != es) return false;

    const size_t e0 = (size_t)std::max<int64_t>(1, t->ne[0]);
    const size_t e1 = (size_t)std::max<int64_t>(1, t->ne[1]);
    const size_t e2 = (size_t)std::max<int64_t>(1, t->ne[2]);

    if (t->nb[1] != e0 * es)       return false;
    if (t->nb[2] != e1 * t->nb[1]) return false;
    if (t->nb[3] != e2 * t->nb[2]) return false;
    return true;
}

bool write_tensor_lightbin(const char* filename, const ggml_tensor* t) {
    if (!t || !t->data) {
        std::fprintf(stderr, "write_tensor_lightbin: null tensor/data\n");
        return false;
    }
    if (!is_c_contiguous(t)) {
        std::fprintf(stderr, "write_tensor_lightbin: tensor is NOT C-contiguous\n");
        return false;
    }

    // map ggml_type -> small integer code
    int32_t dtype_code = -1;
    switch (t->type) {
        case GGML_TYPE_F32:  dtype_code = 0; break;
        case GGML_TYPE_F16:  dtype_code = 1; break;
        case GGML_TYPE_BF16: dtype_code = 2; break;
        case GGML_TYPE_I8:   dtype_code = 3; break;
        case GGML_TYPE_I16:  dtype_code = 4; break;
        case GGML_TYPE_I32:  dtype_code = 5; break;
        case GGML_TYPE_I64:  dtype_code = 6; break;
        // case GGML_TYPE_U8:   dtype_code = 7; break;
        default:
            std::fprintf(stderr, "write_tensor_lightbin: unsupported ggml type %d\n", (int)t->type);
            return false;
    }

    // trim trailing singleton dims (but keep at least 1)
    int32_t num_dims = 4;
    while (num_dims > 1 && t->ne[num_dims - 1] <= 1) num_dims--;

    FILE* f = std::fopen(filename, "wb");
    if (!f) {
        std::fprintf(stderr, "write_tensor_lightbin: fopen('%s') failed: %s\n",
                     filename, std::strerror(errno));
        return false;
    }

    auto fail = [&](const char* msg) {
        std::fprintf(stderr, "write_tensor_lightbin: %s\n", msg);
        std::fclose(f);
        return false;
    };

    // header
    if (std::fwrite(&num_dims, sizeof(num_dims), 1, f) != 1) return fail("write num_dims");
    for (int i = 0; i < num_dims; ++i) {
        int64_t d = t->ne[i];
        if (std::fwrite(&d, sizeof(d), 1, f) != 1) return fail("write dims");
    }
    if (std::fwrite(&dtype_code, sizeof(dtype_code), 1, f) != 1) return fail("write dtype");

    // payload
    const size_t nbytes = ggml_nelements(t) * ggml_element_size(t);
    if (nbytes) {
        if (std::fwrite(t->data, 1, nbytes, f) != nbytes) return fail("write data");
    }

    std::fclose(f);
    return true;
}