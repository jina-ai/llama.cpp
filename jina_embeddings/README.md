# Mteb eval
```bash
python eval_mteb.py \
	--llama-bin /home/andrei/workspace/llama.cpp/build/bin/llama-server \
	--model /home/andrei/workspace/gguf/jev4-bf16.gguf \
	--mmproj /home/andrei/workspace/gguf/mmproj-jev4-bf16.gguf \
	--tasks VidoreSyntheticDocQAEnergyRetrieval \
	--output-dir /home/andrei/workspace/gguf/vidore/jev4-bf16 \
	--gpus 0 \
	--no-logging \
	--query-prefix "Query: " \
	--document-prefix "Passage: " \
	--image-prefix '<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n'
```

Mteb script also supports Vidore task.
To get the name of Vidore tasks, run:
```bash
python -c "import mteb; tasks = mteb.get_tasks(); print([t.metadata.name for t in tasks if 'chart' in t.metadata.name.lower() or 'vidore' in t.metadata.name.lower()])"
```

# Inference example
```bash
python infer.py   \
    --llama-bin /home/andrei/workspace/llama.cpp/build/bin/llama-server   \
    --model /home/andrei/workspace/gguf/jev4-bf16.gguf   \
    --mmproj /home/andrei/workspace/gguf/mmproj-jev4-bf16.gguf   \
    --gpus 1   \
    --input /home/andrei/workspace/test_data.txt   \
    --output /home/andrei/workspace/jev4_mmtd.json   \
    --save-cosine-sim-path /home/andrei/workspace/jev4_mmtd.md   \
    --query-prefix "Query: "   \
    --document-prefix "Passage: "   \
    --normalize-after-pooling
```

# Quantization

## Build importance matrix data
```bash
python build_i_matrix_data.py \
    -f /shared/datasets/text-embedding-training/en/triplets/msmarco-full \
    -f /shared/datasets/text-embedding-training/en/triplets/nq-bge \
    -f /shared/datasets/text-embedding-training/zh/triplets/mmarco-mined-from-pair-random \
    -f /shared/datasets/text-embedding-training/de/triplets/hotpotqa \
    -f /shared/datasets/text-embedding-training/en/triplets/nli-random \
    -f /shared/datasets/text-embedding-training/ja/triplets/nli-random \
    -f /shared/datasets/text-embedding-training/ar/triplets/nli-random \
    -f /shared/datasets/text-embedding-training/fr/triplets/nli-random \
    -f /shared/datasets/text-embedding-training/de/triplets/nli-random \
    -f /shared/datasets/text-embedding-training/es/triplets/nli-random \
    -f /shared/datasets/text-embedding-training/ru/triplets/nli-random \
    -f /shared/datasets/text-embedding-training/zh/triplets/t2ranking-provided-hard \
    -f /shared/datasets/text-embedding-training/en/triplets/fever-mixed \
    -f /shared/datasets/text-embedding-training/zh/triplets/msmarco-bge \
    -f /shared/datasets/text-embedding-training/en/triplets/hotpotqa-mixed \
    -f /shared/datasets/text-embedding-training/zh/triplets/nli-lcqmc-random-v2 \
    -f /shared/datasets/text-embedding-training/zh/triplets/cmedqa2-hard \
    -f /shared/datasets/text-embedding-training/es/triplets/hotpotqa \
    -f /shared/datasets/text-embedding-training/de/triplets/ger-da-lir-jina \
    -f /shared/datasets/text-embedding-training/de/triplets/msmarco-bge \
    -f /shared/datasets/text-embedding-training/es/triplets/msmarco-bge \
    -f /shared/datasets/text-embedding-training/en/triplets/pubmedqa-bm25 \
    -f /shared/datasets/text-embedding-training/en/triplets/fiqa-mixed \
    -f /shared/datasets/text-embedding-training/ja/triplets/msmarco-full \
    -f /shared/datasets/text-embedding-training/ar/triplets/msmarco-full \
    -f /shared/datasets/text-embedding-training/ru/triplets/msmarco-full \
    -f /shared/datasets/text-embedding-training/extra/mlqa-translate \
    --left-prefix "" \
    --right-prefix "" \
    -s 150 \
	--remove-stopwords \
	--remove-punctuation \
    --scramble-method light \
    -o /home/andrei/workspace/retrieval_data_examples__wo_punct_stop_words.txt
```