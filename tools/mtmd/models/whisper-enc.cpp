#include "models.h"

ggml_cgraph * clip_graph_whisper_enc::build() {
    const int n_frames = img.nx;
    const int n_pos    = n_frames / 2;
    GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

    ggml_tensor * inp = build_inp_raw(1);

    // conv1d block. For QWEN2A, conv1+conv2 must run per-chunk (each chunk is
    // zero-padded at its own start/end), matching torch's chunked
    // Qwen2_5OmniAudioEncoder. Other audio models keep the original
    // single-sequence path.
    if (proj_type == PROJECTOR_TYPE_QWEN2A) {
        constexpr int kQwen2aWindowPre = 200; // n_window * 2 mel frames per chunk
        const int n_chunks = n_frames / kQwen2aWindowPre;
        GGML_ASSERT(n_chunks * kQwen2aWindowPre == n_frames &&
                    "n_frames must be a multiple of 200 for chunked Qwen2A audio");

        // For variable-length: if the last chunk has only `real_in_last_chunk`
        // real mel frames (out of kQwen2aWindowPre), torch's `padded_mask`
        // zeros the conv1 output at the invalid positions before conv2. We
        // mirror that here via a graph-input mask we multiply the last
        // partial chunk's conv1 output by.
        const int real_n_frames = (img.nx_real > 0) ? img.nx_real : n_frames;
        const int real_in_last_chunk = real_n_frames - (n_chunks - 1) * kQwen2aWindowPre;
        const bool has_partial_last = (real_in_last_chunk > 0 && real_in_last_chunk < kQwen2aWindowPre);
        ggml_tensor * partial_chunk_mask = nullptr;
        if (has_partial_last) {
            // Shape (kQwen2aWindowPre, 1, 1) so it broadcasts over the
            // n_embd and (single) batch dims of the chunk's conv1 output.
            partial_chunk_mask = ggml_new_tensor_3d(
                ctx0, GGML_TYPE_F32, kQwen2aWindowPre, 1, 1
            );
            ggml_set_name(partial_chunk_mask, "audio_partial_chunk_mask");
            ggml_set_input(partial_chunk_mask);
        }

        // Apply conv1+conv2 to each chunk independently and concatenate the
        // outputs along the length dimension.
        ggml_tensor * concat_out = nullptr;
        for (int c = 0; c < n_chunks; ++c) {
            // View of inp[c*200 : (c+1)*200, :, :] — shape (200, n_mel, 1).
            ggml_tensor * chunk_view = ggml_view_3d(
                ctx0, inp,
                kQwen2aWindowPre, inp->ne[1], inp->ne[2],
                inp->nb[1], inp->nb[2],
                static_cast<size_t>(c) * kQwen2aWindowPre * inp->nb[0]
            );
            ggml_tensor * chunk = ggml_cont(ctx0, chunk_view);

            chunk = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, chunk, 1, 1);
            chunk = ggml_add(ctx0, chunk, model.conv1d_1_b);
            chunk = ggml_gelu_erf(ctx0, chunk);

            // Match torch's `padded_embed = gelu(conv1(...)) * padded_mask`:
            // zero conv1 output at invalid positions of the last partial chunk.
            if (has_partial_last && c == n_chunks - 1) {
                chunk = ggml_mul(ctx0, chunk, partial_chunk_mask);
            }

            chunk = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, chunk, 2, 1);
            chunk = ggml_add(ctx0, chunk, model.conv1d_2_b);
            chunk = ggml_gelu_erf(ctx0, chunk);
            // chunk shape now: (100, n_embd, 1)

            if (concat_out == nullptr) {
                concat_out = chunk;
            } else {
                // Concatenate along ne[0] (length).
                concat_out = ggml_concat(ctx0, concat_out, chunk, 0);
            }
        }
        // concat_out shape: (n_pos = n_chunks * 100, n_embd, 1).

        // Transpose to (n_embd, n_pos, 1).
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, concat_out));
        cb(inp, "after_conv1d_chunked", -1);
    } else {
        // single-sequence convolution + gelu (original Whisper path)
        ggml_tensor * cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_1_b);

        cur = ggml_gelu_erf(ctx0, cur);

        cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_2_b);

        cur = ggml_gelu_erf(ctx0, cur);
        // transpose
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cb(inp, "after_conv1d", -1);
    }

    // sanity check (only check one layer, but it should be the same for all)
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);
    GGML_ASSERT(!model.layers[0].k_b); // no bias for k

    // Qwen2.5-Omni audio uses block-diagonal (chunked) self-attention with chunks
    // of n_window*2 mel frames (= n_window after conv2 stride 2). Create a
    // graph-input mask here; values are filled in clip_image_encode just before
    // ggml_backend_graph_compute. Other audio projector types use full attention
    // (mask stays nullptr).
    //
    // Additionally, torch applies positional embeddings [0..n_window-1] to EACH
    // chunk independently — not as a contiguous [0..n_pos-1] across the full
    // sequence. So we tile the first n_window positions across all chunks for
    // QWEN2A.
    constexpr int kQwen2aWindowPost = 100; // n_window after conv2 stride 2

    ggml_tensor * kq_mask = nullptr;
    ggml_tensor * pos_embd_selected;
    if (proj_type == PROJECTOR_TYPE_QWEN2A) {
        GGML_ASSERT(n_pos % kQwen2aWindowPost == 0 &&
                    "n_pos must be a multiple of n_window for chunked attention");

        // F16 mask required by ggml_flash_attn_ext.
        kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pos, n_pos);
        ggml_set_name(kq_mask, "audio_chunk_mask");
        ggml_set_input(kq_mask);

        // Tile pos_embd[0..n_window-1] across the full n_pos sequence.
        ggml_tensor * pos_embd_chunk = ggml_view_2d(
            ctx0, model.position_embeddings,
            model.position_embeddings->ne[0], kQwen2aWindowPost,
            model.position_embeddings->nb[1], 0
        );
        pos_embd_selected = ggml_repeat_4d(
            ctx0, pos_embd_chunk,
            model.position_embeddings->ne[0], n_pos, 1, 1
        );
    } else {
        pos_embd_selected = ggml_view_2d(
            ctx0, model.position_embeddings,
            model.position_embeddings->ne[0], n_pos,
            model.position_embeddings->nb[1], 0
        );
    }

    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            pos_embd_selected,
                            nullptr,
                            kq_mask);

    cb(cur, "after_transformer", -1);

    // Variable-length output for QWEN2A: truncate the after-avg-pool sequence to
    // the real audio's audio-token count. img.nx_real (if > 0) is the real
    // mel-frame count; the matching audio-token count is floor((nx_real+1)/4).
    if (proj_type == PROJECTOR_TYPE_QWEN2A && img.nx_real > 0) {
        const int real_audio_tokens = (img.nx_real + 1) / 4;
        if (real_audio_tokens > 0 && real_audio_tokens < cur->ne[1]) {
            cur = ggml_view_2d(
                ctx0, cur,
                cur->ne[0], real_audio_tokens,
                cur->nb[1], 0
            );
            cur = ggml_cont(ctx0, cur);
            cb(cur, "after_truncate", -1);
        }
    }

    if (model.audio_has_stack_frames()) {
        // StackAudioFrames
        // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
        cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
        cb(cur, "after_stacked", -1);
    }

    if (proj_type == PROJECTOR_TYPE_ULTRAVOX) {
        // UltravoxProjector
        // pre-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);

        // ffn in
        cur = build_mm(model.mm_1_w, cur);

        // swiglu
        // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
        cur = ggml_swiglu_swapped(ctx0, cur);

        // mid-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_mid_w);

        // ffn out
        cur = build_mm(model.mm_2_w, cur);

    } else if (proj_type == PROJECTOR_TYPE_QWEN2A) {
        // projector
        cur = build_mm(model.mm_fc_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_fc_b);

    } else if (proj_type == PROJECTOR_TYPE_VOXTRAL) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MUSIC_FLAMINGO) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MERALION) {
        // stack (above) -> ln -> linear0+silu -> GLU -> out
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
        cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);

        cur = ggml_mul_mat(ctx0, model.mm_0_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_0_b);
        cur = ggml_silu(ctx0, cur);

        ggml_tensor * gate = ggml_mul_mat(ctx0, model.mm_1_w, cur);
        gate = ggml_add(ctx0, gate, model.mm_1_b);
        gate = ggml_silu(ctx0, gate);

        ggml_tensor * pool = ggml_mul_mat(ctx0, model.mm_2_w, cur);
        pool = ggml_add(ctx0, pool, model.mm_2_b);

        cur = ggml_mul(ctx0, gate, pool);

        cur = ggml_mul_mat(ctx0, model.mm_3_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_3_b);

    } else if (proj_type == PROJECTOR_TYPE_GLMA) {
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
            cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);
            cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
            cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_2_w, model.mm_2_b, hparams.ffn_op, 0);
            cur = ggml_concat(ctx0, model.mm_boi, cur, 1);
            cur = ggml_concat(ctx0, cur, model.mm_eoi, 1);
    } else {
        GGML_ABORT("%s: unknown projector type", __func__);
    }

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
