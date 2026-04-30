# Jina v5 Omni — `feat-v5-omni`

This branch adds the patches required to run Jina's v5 multimodal embedding
models on llama.cpp with full numerical parity vs the torch reference:

- [`jinaai/jina-embeddings-v5-omni-small-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval)
- [`jinaai/jina-embeddings-v5-omni-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-retrieval)

Text, image, audio, video, and PDF are all supported with full numerical
parity vs torch (cos ≥ 0.99). The audio path uses the Jina-specific audio
mmproj (separate from the vision mmproj) — the encoder is a Whisper-style
`Qwen2_5OmniAudioEncoder` (32 layers, d_model=1280, 128 mel bins) with a
linear projector to the text hidden dim. Video uses the standard vision
mmproj with native 3D-conv (temporal_patch_size=2) support — see the
"Multimodal (text + video)" section below for the `videopair_data` API.

## Parity claim

End-to-end embedding cosine similarity vs torch reference (F16 GGUF text +
F16 mmprojs). Numerical-parity bar is **cos ≥ 0.99**.

| Probe | small | nano |
|---|---|---|
| Text — 7 inputs (English + multilingual + long pangram) | 7/7 PASS, min 0.999877 | 6/7 PASS, min 0.998456 (`Bonjour le monde` — accepted bf16→f16 noise) |
| Image `img_car` (640×480) | **0.9989** | **0.9994** |
| Image `img_cat` (32×32, requires 16× upscale) | **0.9990** | **0.9986** |
| Image–query Spearman | 1.0 | 1.0 |
| Audio (JFK speech, 11s @ 16kHz mono) | **0.9997** | **0.9999** |
| Audio–query Spearman | 1.0 | 1.0 |
| PDF (2-page fused image embedding) | **0.9943** | 0.9877 |
| Video (4-frame, 2 logical frames @ 512×512, T=2) | **0.9978** | n/a (nano `LlavaEuroBertAudioConfig` has no `video_token_id`) |
| Video (2-frame, 1 logical frame @ 512×512, T=1) | **0.9981** | n/a |
| Cross-backend query cos | ≥ 0.99994 | ≥ 0.99996 |

Reports: `outputs/gguf-omni-{small,nano}-retrieval/{vision,audio,pdf,video}_parity_report_*_2026-04-30.json`
in the [multimodal-large-scale-training](https://github.com/jina-ai/multimodal-large-scale-training)
repo.

## Patches in this branch

```
e192d1b mtmd: native Qwen3VL video parity via temporal-pair patch_embed
ff59ead mtmd: gate Qwen2.5-Omni variable-length audio behind MTMD_QWEN2A_VARLEN env var
25b6c8b mtmd: full Qwen2.5-Omni audio parity vs torch (chunked attn + per-chunk conv + variable-length)
02c8c1e convert: register audio-only Qwen2.5-Omni mmproj converter
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
  llama-server multi-decode path drops image/audio features. This patch routes
  text+media for embedding models through a single unified embedding batch.
- **`4a252ec` pos_embed interpolation alignment** — switches qwen3vl pos_embed
  resampling from `BILINEAR | ANTIALIAS` to `BILINEAR | ALIGN_CORNERS` to
  match torch's `fast_pos_embed_interpolate`.
- **`ba0d398` bicubic-pillow image resize** — `Qwen2VLImageProcessor.resample == 3`
  (`PIL.BICUBIC`); the default `RESIZE_ALGO_BILINEAR` diverges on heavy upscale.
- **`25b6c8b` Qwen2.5-Omni audio parity** — the audio encoder needed five fixes
  to match torch:
  1. v5-omni audio token names (`<|audio_start|>` / `<|audio_end|>`, since
     `<|audio_bos|>` / `<|audio_eos|>` aren't single tokens in v5-omni's vocab);
  2. block-diagonal chunked self-attention with `n_window=100`-token chunks
     (torch's `Qwen2_5OmniAudioEncoder` chunks the input);
  3. tiled positional embedding (`pos_embd[0..99]` repeated per chunk, not
     contiguous `pos_embd[0..n_pos-1]`);
  4. per-chunk conv1+conv2 (each chunk independently zero-padded at start/end),
     instead of single-sequence conv;
  5. variable-length audio: emit a single mel chunk of `pad_to_200(real_mel_frames)`
     with the tail zeroed; partial-chunk conv1 output mask + extended attention
     mask + view truncation so the encoder emits the same token count as torch's
     natural masked path (e.g. 275 audio tokens for 11s, not the previous
     fixed 750).

The other patches (`fa8376b`, `0b9cf28`, `37d0f04`, `5df11fb`, `02c8c1e`) cover
model-loading compatibility (optional patch_bias, projection_dim from tensor
shape, F16 window mask, `n_wa_pattern` gating, vision-token vocab guard,
audio-only Qwen2.5-Omni mmproj registration) and runtime metadata plumbing
(`image_min_pixels` / `image_max_pixels` from the HF preprocessor config).

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

### Multimodal (text + video)

Video uses the same vision mmproj as image, but the qwen3vl encoder graph
runs in temporal-pair mode (`temporal_patch_size=2`): each `<__media__>`
marker consumes a frame *pair*, with `patch_embeddings_0` acting on
frame_a and `patch_embeddings_1` acting on frame_b — exactly mirroring
torch's 3D conv with `kt=2`.

Caller workflow (see `scripts/omni/gguf/test_gguf_video_parity.py` in
the multimodal-large-scale-training repo for a reference harness):

1. Use Qwen3VLProcessor on the python side to discover the canonical
   prompt structure (timestamp text + N vision blocks for the video).
2. Decode video frames; group consecutive pairs (matching
   `temporal_patch_size=2`).
3. Build the prompt as literal timestamp text + one `<__media__>`
   marker per logical frame.
4. POST to `/embeddings` using the new `videopair_data` field (parallel
   to `multimodal_data`); each entry is a 2-element array of
   base64-encoded image bytes (frame_a, frame_b).

```json
{
  "content": [
    {
      "prompt_string": "<0.2 seconds><__media__><1.2 seconds><__media__>",
      "videopair_data": [
        ["<base64-frame-a-pair-0>", "<base64-frame-b-pair-0>"],
        ["<base64-frame-a-pair-1>", "<base64-frame-b-pair-1>"]
      ]
    }
  ]
}
```

Both frames in each pair must decode to the same dimensions; the same
image preprocessor runs on each frame, then the encoder graph
sums the two convs. The fixed-size `image_min_pixels` constraint from
the vision mmproj's metadata still applies, so frame dimensions should
be ≥ the encoder's minimum (default 512² for v5-omni's vision mmproj)
to avoid resize divergence vs torch's video processor (which uses its
own native sizing).

Multimodal_data and videopair_data can also be mixed in a single
request — each marker consumes one entry from `multimodal_data` (single
image or audio) or one pair from `videopair_data` (video pair), in the
order they appear in the prompt; the server routes by media type.

### Multimodal (text + audio)

```bash
./build/bin/llama-server \
  -m jina-embeddings-v5-omni-small-retrieval-F16.gguf \
  --mmproj jina-embeddings-v5-omni-small-retrieval-audio-mmproj-F16.gguf \
  --embedding --pooling last \
  --host 127.0.0.1 --port 8080 \
  -b 4096 -ub 4096
```

The audio mmproj is a separate file from the vision mmproj. To use both
modalities at once you can serve two endpoints (one per mmproj) — the
runtime currently loads at most one mmproj per server process.

The `-b 4096 -ub 4096` flags bump the physical batch size since audio
prompts can expand to ~750 tokens for a 30s clip (the audio path inserts
one placeholder token per audio embedding).

POST to `/embeddings` with the v5-omni audio prompt template and a base64
audio payload (WAV/MP3/FLAC). The runtime picks the audio length from
the bytes — for an 11s clip it emits 275 audio tokens; for a 30s clip,
750.

```json
{
  "content": [
    {
      "prompt_string": "<__media__>",
      "multimodal_data": ["<base64-encoded-audio-bytes>"]
    }
  ]
}
```

The `<__media__>` placeholder is replaced server-side with the right
sequence of audio tokens (wrapped in `<|audio_start|>` / `<|audio_end|>`
boundary tokens). Audio is resampled to 16kHz mono internally; supported
input formats are WAV, MP3, and FLAC (via `miniaudio`).

## Notes

- This is a private fork tracking upstream `ggml-org/llama.cpp`. The patches
  here are not (yet) submitted upstream.
- The vision mmprojs released alongside v5-omni include `image_min_pixels` /
  `image_max_pixels` metadata; older vision mmprojs converted before commit
  `5df11fb` need patching with `scripts/omni/gguf/diag_patch_mmproj_pixels.py`
  from the multimodal-large-scale-training repo.
- The audio mmproj's `n_window` is hardcoded to 100 in C++ for now (matching
  `Qwen2_5OmniAudioEncoderConfig.n_window`). If a future model uses a different
  window size it would need to be wired through gguf metadata; the converter
  already writes `audio_config` keys, just not `n_window` yet.
- See [`AGENTS.md`](AGENTS.md) for the upstream contribution policy. Private
  forks (this one) are exempt.
