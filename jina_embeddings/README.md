# Jina Embeddings v4 (GGUF)

This repository provides **GGUF-converted builds** of [Jina AI’s v4 embeddings](https://huggingface.co/jinaai/jina-embeddings-v4) for use with [`llama.cpp`](https://github.com/ggml-org/llama.cpp).

It supports **text + image retrieval tasks**, evaluation with [MTEB](https://github.com/embeddings-benchmark/mteb), and running your own embedding service.

---

## Installation

### Linux (CUDA)

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9
cmake --build build --config Release
```

### macOS (Metal)

```bash
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_METAL=ON
cmake --build build --config Release
```

---

## Model files

Download with [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start):

```bash
cd llama.cpp
huggingface-cli download jinaai/jina-embeddings-v4-text-retrieval-GGUF \
  --include jina-embeddings-v4-text-retrieval-F16.gguf \
  --local-dir gguf-models

huggingface-cli download jinaai/jina-embeddings-v4-text-retrieval-GGUF \
  --include mmproj-jina-embeddings-v4-retrieval-BF16.gguf \
  --local-dir gguf-models
```

👉 Use the **BF16 mmproj**. The F16 mmproj can produce NaN values in some benchmarks.

---

## MTEB evaluation (Vidore tasks)

```bash
cd llama.cpp
python jina_embeddings/eval_mteb.py \
  --llama-bin build/bin/llama-server \
  --model "$PWD/gguf-models/jina-embeddings-v4-text-retrieval-F16.gguf" \
  --mmproj "$PWD/gguf-models/mmproj-jina-embeddings-v4-retrieval-BF16.gguf" \
  --tasks VidoreTatdqaRetrieval \
  --output-dir jev4-gguf-vidore \
  --gpus 0 \
  --no-logging \
  --query-prefix "Query: " \
  --document-prefix "Passage: " \
  --image-prefix '<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n'
```

List available Vidore tasks:

```bash
python -c "import mteb; tasks = mteb.get_tasks(); print([t.metadata.name for t in tasks if 'chart' in t.metadata.name.lower() or 'vidore' in t.metadata.name.lower()])"
```

---

## Inference

### Single image -> `.npy`

```bash
cd llama.cpp
python jina_embeddings/infer_image.py \
  --llama-bin build/bin/llama-server \
  --model "$PWD/gguf-models/jina-embeddings-v4-text-retrieval-F16.gguf" \
  --mmproj "$PWD/gguf-models/mmproj-jina-embeddings-v4-retrieval-BF16.gguf" \
  --hf-model-name "jinaai/jina-embeddings-v4" \
  --output-base jina_embeddings/temp/saved_embeddings \
  --gpus 0 \
  jina_embeddings/assets/jina_embeddings_v4_perf_table.jpg
```

This runs `llama-server`, computes an embedding for the image, and saves it as a `.npy` file.
Embeddings are **pooled and L2-normalized** by default.

---

## Conversion (HF → GGUF)

```bash
huggingface-cli download jinaai/jev4-retrieval --local-dir jev4-retrieval

mkdir -p gguf-models

python convert_hf_to_gguf.py jev4-retrieval \
  --outfile gguf-models/jina-embeddings-v4-text-retrieval-F16.gguf \
  --outtype f16

python convert_hf_to_gguf.py jev4-retrieval \
  --outfile gguf-models/jina-embeddings-v4-BF16.gguf \
  --outtype bf16 \
  --mmproj
```

---

## Build your own service

You can integrate the embeddings into any app by subclassing `LlamaCppServerEmbeddingModel`.
Always set:

* `pool=True` -> ensures a single vector per input (pooled output)
* `normalize=True` -> applies L2 normalization for cosine similarity

Use the correct prefixes.
For the text/image retrieval model, make sure to use:
* **Queries:** `Query: `
* **Documents:** `Passage: `
* **Images:** `<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n`

### Minimal example

```python
from model import LlamaCppServerEmbeddingModel, EmbeddingRequestItem

class MyEmbeddingService(LlamaCppServerEmbeddingModel):
    def __init__(self, **kw):
        super().__init__(pool=True, normalize=True, **kw)
        self.qp = "Query: "
        self.dp = "Passage: "
        self.ip = "<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n"

    def embed_texts(self, texts, as_queries=False):
        p = self.qp if as_queries else self.dp
        items = [EmbeddingRequestItem(content=p + t, image=None) for t in texts]
        return self.encode(items).tolist()

    def embed_images(self, paths_or_pils):
        items = [EmbeddingRequestItem(content=self.ip, image=i) for i in paths_or_pils]
        return self.encode(items).tolist()
```

This gives you a clean service interface: `embed_texts()` and `embed_images()`.
Extend it with batching, HTTP endpoints, or queueing as needed.