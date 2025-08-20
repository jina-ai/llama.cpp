# Installation

Compile with cURL - recommended.
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9
cmake --build build --config Release
```

Compile without cURL - doing this where we don't have cURL installed.
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9 -DLLAMA_CURL=OFF
cmake --build build --config Release
```

Compile on Mac (gpu).
```bash
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_METAL=ON
cmake --build build --config Release -j 
```

# Mteb eval
You can donwload model files and mmproj like so:
```
cd llama.cpp
huggingface-cli download jinaai/jina-embeddings-v4-text-retrieval-GGUF --include jina-embeddings-v4-text-retrieval-F16.gguf --local-dir gguf-models
huggingface-cli download jinaai/jina-embeddings-v4-text-retrieval-GGUF --include mmproj-jina-embeddings-v4-retrieval-BF16.gguf --local-dir gguf-models
```
We recommend using the BF16 mmproj file since currently there seems to be a problem with the F16 mmproj that produces NaN embeddings on some Vidore benchmarks.

```bash
cd llama.cpp
python jina_embeddings/eval_mteb.py \
	--llama-bin build/bin/llama-server \
	--model "$PWD/gguf-models/jina-embeddings-v4-text-retrieval-F16.gguf" \
	--mmproj "$PWD/gguf-models/mmproj-jina-embeddings-v4-retrieval-BF16.gguf" \
	--tasks VidoreSyntheticDocQAEnergyRetrieval \
	--output-dir jev4-gguf-vidore \
	--gpus 0 \
	--no-logging \
	--query-prefix "Query: " \
	--document-prefix "Passage: " \
	--image-prefix '<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n'
```

To get the name of all Vidore tasks, run:
```bash
python -c "import mteb; tasks = mteb.get_tasks(); print([t.metadata.name for t in tasks if 'chart' in t.metadata.name.lower() or 'vidore' in t.metadata.name.lower()])"
```

# Inference example
```bash
cd llama.cpp
python jina_embeddings/infer_cosine.py   \
    --llama-bin llama.cpp/build/bin/llama-server   \
	--model gguf-models/jev4-bf16.gguf \
    --mmproj gguf-models/mmproj-jina-embeddings-v4-retrieval-BF16.gguf \
    --gpus 0 \
    --input jina_embeddings/assets/test_data.txt   \
    --output jev4_cosine_results.md   \
    --query-prefix "Query: "   \
    --document-prefix "Passage: "   \
    --normalize
```

# Conversion

```bash
huggingface-cli download jinaai/jev4-retrieval --local-dir jev4-retrieval

mkdir gguf-models

python convert_hf_to_gguf.py jev4-retrieval \
    --outfile gguf-models/jina-embeddings-v4-text-retrieval-F16.gguf \
    --outtype f16 

python convert_hf_to_gguf.py jev4-retrieval \
    --outfile gguf-models/jina-embeddings-v4-F16.gguf \
    --outtype f16 \
    --mmproj
```