# Inference example
```bash
python infer.py   \
    --llama-bin /home/andrei/workspace/llama.cpp/build/bin/llama-server   \
    --model /home/andrei/workspace/gguf/jev4-bf16.gguf   \
    --mmproj /home/andrei/workspace/gguf/mmproj-jev4-bf16.gguf   \
    --gpus 7   \
    --input /home/andrei/workspace/test_data.txt   \
    --output /home/andrei/workspace/jev4_mmtd.json   \
    --save-cosine-sim-path /home/andrei/workspace/jev4_mmtd.md   \
    --query-prefix "Query: "   \
    --document-prefix "Passage: "   \
    --normalize-after-pooling
```