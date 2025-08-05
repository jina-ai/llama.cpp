#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <vector>
#include "ggml.h"

void load_tensor_from_file(struct ggml_tensor* tensor, const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filename);
        exit(1);
    }
    
    // Calculate total elements
    size_t total_elements = ggml_nelements(tensor);
    
    // Read flattened data from file
    std::vector<float> flat_data(total_elements);
    for (size_t i = 0; i < total_elements; i++) {
        if (fscanf(f, "%f", &flat_data[i]) != 1) {
            fprintf(stderr, "ERROR: Failed to read element %zu from %s\n", i, filename);
            exit(1);
        }
    }
    fclose(f);
    
    // Now we need to map from PyTorch's row-major flattened order to GGML's layout
    float* tensor_data = (float*)tensor->data;
    
    if (tensor->ne[2] == 0) {
        // 2D tensor: cos/sin [d_head, n_patch] in GGML vs [n_patch, d_head] in PyTorch
        const int d_head = tensor->ne[0];    // 80
        const int n_patch = tensor->ne[1];   // 280
        
        for (int patch = 0; patch < n_patch; patch++) {
            for (int head_dim = 0; head_dim < d_head; head_dim++) {
                // PyTorch flattened index: patch * d_head + head_dim
                size_t pytorch_idx = patch * d_head + head_dim;
                // GGML index: head_dim * n_patch + patch
                size_t ggml_idx = head_dim * n_patch + patch;
                tensor_data[ggml_idx] = flat_data[pytorch_idx];
            }
        }
    } else {
        // 3D tensor: q [d_head, n_head, n_patch] in GGML vs [n_patch, n_head, d_head] in PyTorch
        const int d_head = tensor->ne[0];    // 80
        const int n_head = tensor->ne[1];    // 16  
        const int n_patch = tensor->ne[2];   // 280
        
        for (int patch = 0; patch < n_patch; patch++) {
            for (int head = 0; head < n_head; head++) {
                for (int head_dim = 0; head_dim < d_head; head_dim++) {
                    // PyTorch flattened index: patch * (n_head * d_head) + head * d_head + head_dim
                    size_t pytorch_idx = patch * (n_head * d_head) + head * d_head + head_dim;
                    // GGML index: head_dim * (n_head * n_patch) + head * n_patch + patch
                    size_t ggml_idx = head_dim * (n_head * n_patch) + head * n_patch + patch;
                    tensor_data[ggml_idx] = flat_data[pytorch_idx];
                }
            }
        }
    }
}

void log_to_file_or_console(FILE* output_file, ggml_tensor* t) {
    if (!t) return;
    
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
        
        // Log first 5 patches
        int patches_to_log = std::min(5, n_patch);
        for (int patch = 0; patch < patches_to_log; patch++) {
            PRINT_TO_OUTPUT("Patch %d: ", patch);
            
            // Log first 5 values of d_head dimension
            int values_to_log = std::min(5, d_head);
            for (int head_dim = 0; head_dim < values_to_log; head_dim++) {
                // GGML index: head_dim * n_patch + patch
                size_t ggml_idx = head_dim * n_patch + patch;
                PRINT_TO_OUTPUT("%.6f ", data[ggml_idx]);
            }
            if (d_head > 5) {
                PRINT_TO_OUTPUT("... ");
            }
            PRINT_TO_OUTPUT("\n");
        }
        if (n_patch > 5) {
            PRINT_TO_OUTPUT("...\n");
        }
    } else {
        // 3D tensor: [d_head, n_head, n_patch] in GGML
        const int d_head = t->ne[0];
        const int n_head = t->ne[1];
        const int n_patch = t->ne[2];
        
        // Log first 5 patches
        int patches_to_log = std::min(5, n_patch);
        for (int patch = 0; patch < patches_to_log; patch++) {
            PRINT_TO_OUTPUT("Patch %d\n", patch);
            
            // Log first 4 heads
            int heads_to_log = std::min(4, n_head);
            for (int head = 0; head < heads_to_log; head++) {
                PRINT_TO_OUTPUT("  Head %d: ", head);
                
                // Log first 5 values of d_head dimension for this head
                int values_to_log = std::min(5, d_head);
                for (int head_dim = 0; head_dim < values_to_log; head_dim++) {
                    // GGML index: head_dim * (n_head * n_patch) + head * n_patch + patch
                    size_t ggml_idx = head_dim * (n_head * n_patch) + head * n_patch + patch;
                    PRINT_TO_OUTPUT("%.6f ", data[ggml_idx]);
                }
                if (d_head > 5) {
                    PRINT_TO_OUTPUT("...");
                }
                PRINT_TO_OUTPUT("\n");
            }
            if (n_head > 4) {
                PRINT_TO_OUTPUT("  ...\n");
            }
        }
        if (n_patch > 5) {
            PRINT_TO_OUTPUT("...\n");
        }
    }
    
    PRINT_TO_OUTPUT("\n");
    #undef PRINT_TO_OUTPUT
}


std::vector<std::pair<int, int>> generate_position_ids(
    int grid_h, int grid_w, int spatial_merge_size
) {
    std::vector<std::pair<int, int>> pos_ids;
    
    // For single image (t=1), generate h,w position pairs
    // Following the spatial merge and permutation logic from PyTorch
    
    int h = grid_h;
    int w = grid_w;
    int merge_size = spatial_merge_size;
    
    // Create position grids
    std::vector<std::vector<int>> hpos_ids(h, std::vector<int>(w));
    std::vector<std::vector<int>> wpos_ids(h, std::vector<int>(w));
    
    // Fill position grids
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            hpos_ids[i][j] = i;
            wpos_ids[i][j] = j;
        }
    }
    
    // Reshape and permute following PyTorch logic
    int h_blocks = h / merge_size;
    int w_blocks = w / merge_size;
    
    for (int hb = 0; hb < h_blocks; hb++) {
        for (int wb = 0; wb < w_blocks; wb++) {
            for (int hi = 0; hi < merge_size; hi++) {
                for (int wi = 0; wi < merge_size; wi++) {
                    int h_pos = hpos_ids[hb * merge_size + hi][wb * merge_size + wi];
                    int w_pos = wpos_ids[hb * merge_size + hi][wb * merge_size + wi];
                    pos_ids.push_back({h_pos, w_pos});
                }
            }
        }
    }
    
    return pos_ids;
}

// Compute rotary embeddings for given positions
// Equivalent to Qwen2_5_VisionRotaryEmbedding
void compute_rotary_embeddings(
    const std::vector<std::pair<int, int>>& pos_ids,
    int n_patches,
    int d_head,
    int d_pair,
    float theta_base,
    float* cos_data,
    float* sin_data
) {
    // Compute frequency inverse (inv_freq equivalent)
    std::vector<float> inv_freq(d_pair);
    for (int i = 0; i < d_pair; i++) {
        inv_freq[i] = 1.0f / powf(theta_base, (float)(2 * i) / (float)d_head);
    }
    
    // For each patch position
    for (int patch = 0; patch < n_patches; patch++) {
        int h_pos = pos_ids[patch].first;
        int w_pos = pos_ids[patch].second;
        
        // For each rotation pair
        for (int pair = 0; pair < d_pair; pair++) {
            // CRITICAL DISCOVERY: PyTorch only uses H position, ignores W!
            // This explains why [0,1] gives same result as [0,0]
            float freq = inv_freq[pair];
            float angle = h_pos * freq;  // Only use h_pos!
            
            // GGML tensor layout: [d_head, n_patches] for cos/sin
            int cos_idx = pair * n_patches + patch;
            int sin_idx = pair * n_patches + patch;
            
            cos_data[cos_idx] = cosf(angle);
            sin_data[sin_idx] = sinf(angle);
        }
    }
}

// Main function to generate 3D RoPE embeddings
void generate_3d_rope_embeddings(
    int n_patches,
    int d_head,
    int grid_h,
    int grid_w,
    int spatial_merge_size,
    float theta_base,
    struct ggml_tensor* cos_tensor,
    struct ggml_tensor* sin_tensor
) {
    int d_pair = d_head / 2;
    
    // Verify tensor dimensions
    assert(cos_tensor->ne[0] == d_pair);    // Only d_pair frequencies
    assert(cos_tensor->ne[1] == n_patches);
    assert(sin_tensor->ne[0] == d_pair);
    assert(sin_tensor->ne[1] == n_patches);
    
    float* cos_data = (float*)cos_tensor->data;
    float* sin_data = (float*)sin_tensor->data;
    
    // Generate position IDs following PyTorch logic
    auto pos_ids = generate_position_ids(grid_h, grid_w, spatial_merge_size);
    
    // Compute rotary embeddings
    compute_rotary_embeddings(pos_ids, n_patches, d_head, d_pair, theta_base, cos_data, sin_data);
}

void apply_rotary_embeddinga3d(
    struct ggml_tensor* dst,
    const struct ggml_tensor* cos_data,
    const struct ggml_tensor* sin_data
) {
    // GGML tensor dimensions
    const int d_head = dst->ne[0];      // 80 (head dimension)
    const int n_heads = dst->ne[1];     // 16 (number of heads)
    const int n_patches = dst->ne[2];   // 280 (number of patches/sequence length)
    
    // Validate dimensions
    assert(d_head % 2 == 0);  // Head dimension must be even for pairwise rotation
    assert(cos_data->ne[0] == d_head && cos_data->ne[1] == n_patches);
    assert(sin_data->ne[0] == d_head && sin_data->ne[1] == n_patches);
    
    const int d_pair = d_head / 2;  // Number of rotation pairs (40)
    
    printf("\n=== RoPE Debug Info ===\n");
    printf("d_head: %d, n_heads: %d, n_patches: %d, d_pair: %d\n", 
           d_head, n_heads, n_patches, d_pair);
    
    float* q_data = (float*)dst->data;
    const float* cos_ptr = (const float*)cos_data->data;
    const float* sin_ptr = (const float*)sin_data->data;
    
    // Apply RoPE to each patch and head
    for (int patch = 0; patch < n_patches; ++patch) {
        for (int head = 0; head < n_heads; ++head) {
            
            // Debug: Print BEFORE values for first few patches/heads
            if (patch < 3 && head == 0) {
                printf("BEFORE RoPE - Patch %d, Head %d: ", patch, head);
                for (int i = 0; i < 10 && i < d_head; i++) {
                    // GGML index for q[head_dim=i, head, patch]
                    size_t q_idx = i * (n_heads * n_patches) + head * n_patches + patch;
                    printf("%.6f ", q_data[q_idx]);
                }
                printf("\n");
            }
            
            // Apply rotation to each pair of dimensions
            // Standard RoPE: rotate (i, i+40) for i = 0..39 using cos[i] and sin[i]
            for (int pair = 0; pair < d_pair; ++pair) {
                // Calculate indices for the pair (pair, pair+d_pair)
                int dim0 = pair;           // First half: 0, 1, 2, 3, ..., 39
                int dim1 = pair + d_pair;  // Second half: 40, 41, 42, 43, ..., 79
                
                // GGML indices for q tensor: [d_head, n_heads, n_patches]
                // q[dim, head, patch] -> dim * (n_heads * n_patches) + head * n_patches + patch
                size_t q_idx0 = dim0 * (n_heads * n_patches) + head * n_patches + patch;
                size_t q_idx1 = dim1 * (n_heads * n_patches) + head * n_patches + patch;
                
                // GGML indices for cos/sin tensors: [d_head, n_patches]
                // cos[dim, patch] -> dim * n_patches + patch
                // Use cos[pair] and sin[pair] for both dimensions in the rotation
                size_t cos_idx = pair * n_patches + patch;
                size_t sin_idx = pair * n_patches + patch;
                
                // Debug: Print indexing for first few operations
                if (patch < 3 && head == 0 && pair < 2) {
                    printf("    Index debug: patch=%d, pair=%d, dim0=%d, dim1=%d, cos_idx=%zu\n",
                           patch, pair, dim0, dim1, cos_idx);
                }
                
                // Get the values
                float x0 = q_data[q_idx0];  // q[dim0, head, patch]
                float x1 = q_data[q_idx1];  // q[dim1, head, patch]
                float c = cos_ptr[cos_idx]; // cos[dim0, patch] (same as cos[dim1, patch])
                float s = sin_ptr[sin_idx]; // sin[dim0, patch] (same as sin[dim1, patch])
                
                // Apply RoPE rotation formula:
                // new_x0 = x0 * cos - x1 * sin  (rotate_half switches x1 to -x1)
                // new_x1 = x0 * sin + x1 * cos  (rotate_half switches x0 to x0)
                q_data[q_idx0] = x0 * c - x1 * s;
                q_data[q_idx1] = x0 * s + x1 * c;
                
                // Debug: Print cos/sin values for first few operations
                if (patch < 2 && head == 0 && pair < 3) {
                    printf("  Patch %d, Head %d, Pair %d: x0=%.6f, x1=%.6f, cos=%.6f, sin=%.6f -> new_x0=%.6f, new_x1=%.6f\n",
                           patch, head, pair, x0, x1, c, s, q_data[q_idx0], q_data[q_idx1]);
                }
            }
            
            // Debug: Print AFTER values for first few patches/heads
            if (patch < 3 && head == 0) {
                printf("AFTER RoPE - Patch %d, Head %d: ", patch, head);
                for (int i = 0; i < 10 && i < d_head; i++) {
                    // GGML index for q[head_dim=i, head, patch]
                    size_t q_idx = i * (n_heads * n_patches) + head * n_patches + patch;
                    printf("%.6f ", q_data[q_idx]);
                }
                printf("\n");
            }
        }
    }
    
    printf("=== RoPE Application Complete ===\n");
}

int main() {
    printf("=== 3D RoPE Test with GGML ===\n");

    // 1) GGML context
    struct ggml_init_params params = {
        .mem_size   = 128ull*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context* ctx = ggml_init(params);

    // 2) Sizes
    const int n_patches = 280;
    const int n_heads   = 16;
    const int d_head    = 80;

    // 3) Allocate tensors in GGML layout:
    // GGML: [d_head, n_patches] for cos/sin
    // GGML: [d_head, n_heads, n_patches] for q
    // auto cos_data      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_head, n_patches);
    // auto sin_data      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_head, n_patches);

    // struct ggml_tensor* cos_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 40, 280);  // [d_pair, n_patches]
    // struct ggml_tensor* sin_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 40, 280);

    generate_3d_rope_embeddings(280, 80, 14, 20, 2, 10000.0f, cos_tensor, sin_tensor);

    printf("\n--- cos tensor ---\n");
    log_to_file_or_console(nullptr, cos_tensor, false);
    printf("\n--- sin tensor ---\n");
    log_to_file_or_console(nullptr, sin_tensor, false);

    ggml_free(ctx);
    return 0;

    // auto input_tensor  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, n_heads, n_patches);
    // auto output_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, n_heads, n_patches);

    // ggml_set_name(cos_data,     "cos_data");
    // ggml_set_name(sin_data,     "sin_data");
    // ggml_set_name(input_tensor, "input_tensor");
    // ggml_set_name(output_tensor,"output_tensor");

    // // 4) Load from PyTorch dumps (they were flattened in row-major order)
    // load_tensor_from_file(cos_data,     "/home/andrei/workspace/qwen25_cos.txt");
    // load_tensor_from_file(sin_data,     "/home/andrei/workspace/qwen25_sin.txt");
    // load_tensor_from_file(input_tensor, "/home/andrei/workspace/qwen25_q_tensor.txt");

    // // 5) Initial logs
    // log_to_file_or_console(nullptr, cos_data,     false);
    // log_to_file_or_console(nullptr, sin_data,     false);
    // log_to_file_or_console(nullptr, input_tensor, false);

    // // 6) Copy & apply RoPE
    // memcpy(output_tensor->data, input_tensor->data, ggml_nbytes(input_tensor));
    // printf("\n--- Applying 3D RoPE ---\n");
    // apply_rotary_embedding_3d(output_tensor, cos_data, sin_data);

    // // 7) Final log
    // log_to_file_or_console(nullptr, output_tensor, false);

    // // 8) Cleanup
    // ggml_free(ctx);
    // return 0;
}