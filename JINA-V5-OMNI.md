# Jina v5 Omni — `feat-v5-omni`

This branch adds the patches required to run Jina's v5 multimodal embedding
models on llama.cpp with full numerical parity vs the torch reference:

- [`jinaai/jina-embeddings-v5-omni-small-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval)
- [`jinaai/jina-embeddings-v5-omni-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-retrieval)

Audio is not in scope for this branch.

## Parity claim

End-to-end embedding cosine similarity vs torch reference (F16 GGUF text +
F16 mmproj vision, with metadata fix). Numerical-parity bar is **cos ≥ 0.99**.

| Probe | small | nano |
|---|---|---|
| Text — 7 inputs (English + multilingual + long pangram) | 7/7 PASS, min 0.999877 | 6/7 PASS, min 0.998456 (`Bonjour le monde` — accepted bf16→f16 noise) |
| Image `img_car` (640×480) | **0.9989** | **0.9994** |
| Image `img_cat` (32×32, requires 16× upscale) | **0.9990** | **0.9986** |
| Image–query Spearman | 1.0 | 1.0 |
| Cross-backend query cos | ≥ 0.99994 | ≥ 0.99996 |

Reports: `outputs/gguf-omni-{small,nano}-retrieval/vision_parity_report_canonical_2026-04-30.json`
in the [multimodal-large-scale-training](https://github.com/jina-ai/multimodal-large-scale-training)
repo.

## Patches in this branch

```
ba0d398 mtmd: qwen2vl/2.5vl/3vl use bicubic-pillow to match torch preprocessor
0b9cf28 mtmd: add bilinear-pillow resize algo
5df11fb qwen2vl: also write image_min/max_pixels metadata
37d0f04 qwen3vl: carry image_min_pixels/image_max_pixels through
4a252ec qwen3vl: fix pos_embed interpolation alignment
8127970 mtmd: encoder multimodal combined decode
fa8376b mtmd: add v5 omni qwen2.5 compatibility fixes
```

The semantic load-bearing patches are:

- **`8127970` encoder combined-decode** — encoder-only multimodal models (`res=nullptr`
  archs like EuroBERT/BERT/JinaBERT) have no KV cache, so the baseline
  llama-server multi-decode path drops image features. This patch routes
  text+media for embedding models through a single unified embedding batch.
- **`4a252ec` pos_embed interpolation alignment** — switches qwen3vl pos_embed
  resampling from `BILINEAR | ANTIALIAS` to `BILINEAR | ALIGN_CORNERS` to
  match torch's `fast_pos_embed_interpolate`.
- **`ba0d398` bicubic-pillow image resize** — `Qwen2VLImageProcessor.resample == 3`
  (`PIL.BICUBIC`); the default `RESIZE_ALGO_BILINEAR` diverges on heavy upscale.

The other patches (`fa8376b`, `0b9cf28`, `37d0f04`, `5df11fb`) cover model-loading
compatibility (optional patch_bias, projection_dim from tensor shape, F16 window
mask, `n_wa_pattern` gating, vision-token vocab guard) and runtime metadata
plumbing (`image_min_pixels` / `image_max_pixels` from the HF preprocessor config).

## Building

Standard CMake build, as upstream:

```bash
git clone https://github.com/jina-ai/llama.cpp.git
cd llama.cpp
git checkout feat-v5-omni

cmake -B build
cmake --build build --config Release -j
```

For CUDA:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

After build, you will need:

- `build/bin/llama-server` — for embedding endpoint
- `build/bin/llama-embedding` — for CLI embedding
- `build/bin/llama-mtmd-cli` — for multimodal CLI

## Running

### Text embedding

```bash
./build/bin/llama-server \
  -m jina-embeddings-v5-omni-small-retrieval-F16.gguf \
  --embedding --pooling last \
  --host 127.0.0.1 --port 8080
```

POST to `/embeddings`:

```json
{ "content": [ { "prompt_string": "Query: what is machine learning?" } ] }
```

### Multimodal (text + image)

```bash
./build/bin/llama-server \
  -m jina-embeddings-v5-omni-small-retrieval-F16.gguf \
  --mmproj jina-embeddings-v5-omni-small-retrieval-vision-mmproj-F16.gguf \
  --embedding --pooling last \
  --image-min-tokens 256 \
  --host 127.0.0.1 --port 8080
```

For nano, additionally pin a fixed token budget:

```bash
  --image-min-tokens 256 --image-max-tokens 256
```

POST to `/embeddings` with the v5-omni image prompt template and a base64
image payload:

```json
{
  "content": [
    {
      "prompt_string": "<|vision_start|><|image_pad|><|vision_end|>",
      "multimodal_data": ["<base64-encoded-image-bytes>"]
    }
  ]
}
```

## Notes

- This is a private fork tracking upstream `ggml-org/llama.cpp`. The patches
  here are not (yet) submitted upstream.
- The mmprojs released alongside v5-omni include `image_min_pixels` /
  `image_max_pixels` metadata; older mmprojs converted before commit `5df11fb`
  need patching with `scripts/omni/gguf/diag_patch_mmproj_pixels.py` from the
  multimodal-large-scale-training repo.
- See [`AGENTS.md`](AGENTS.md) for the upstream contribution policy. Private
  forks (this one) are exempt.
