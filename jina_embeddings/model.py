import base64
import io
import os
import signal
import subprocess
import time
from typing import List, Optional, Union

import numpy as np # type: ignore
import requests # type: ignore
from PIL import Image # type: ignore
from typing_extensions import TypedDict # type: ignore
from tqdm import tqdm # type: ignore
from transformers import AutoTokenizer # type: ignore


class EmbeddingRequestItem(TypedDict):
    content: str
    image: Optional[Union[str, Image.Image]]


class LlamaCppServerEmbeddingModel:
    def __init__(
        self, 
        llama_bin: str,
        model_path: str,
        mmproj_path: str,
        port: int = 8080,
        host: str = "0.0.0.0",
        ngl: int = 999,
        gpus: str = "0",
        ctx_size: int = 4096,
        ubatch_size: int = 4096,
        normalize: bool = False, 
        logging: bool = True,
        hf_tokenizer_name: Optional[str] = None,
        max_text_length: int = 512
    ) -> None:
        self.llama_bin = llama_bin
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.port = port
        self.host = host
        self.ngl = ngl
        self.gpus = gpus
        self.ctx_size = ctx_size
        self.ubatch_size = ubatch_size
        self.normalize = normalize
        self.logging = logging
        self.hf_tokenizer_name = hf_tokenizer_name
        self.max_text_length = max_text_length
        self.server_process = None
        self.hf_tokenizer = None
        
        # Set server URL
        self.server_url = f"http://{host}:{port}"
        
        # Load tokenizer if specified
        if self.hf_tokenizer_name is not None:
            self._log(f"Loading HuggingFace tokenizer: {self.hf_tokenizer_name}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_name, use_fast=True)
        
        # Start server
        self._start_server()
        self._wait_for_server()

    def _log(self, message: str) -> None:
        """Log message if logging is enabled"""
        if self.logging:
            print(message)

    def _start_server(self) -> None:
        """Start the llama-server process"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.gpus

        cmd = [
            self.llama_bin,
            '-m', self.model_path,
            '--mmproj', self.mmproj_path,
            '--embedding',
            '--port', str(self.port),
            '-ngl', str(self.ngl),
            '--host', self.host,
            '--pooling', 'none',
            '--ctx-size', str(self.ctx_size),
            '--ubatch-size', str(self.ubatch_size)
        ]
        
        self._log(f"Starting llama-server with: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(cmd, env=env)

    def _wait_for_server(self, max_wait_time: int = 300, check_interval: int = 2) -> None:
        """Wait for the server to be ready"""
        self._log("Waiting for server to start...")
        test_payload = {"content": "test"}

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise TimeoutError(f"Server did not become ready within {max_wait_time} seconds")
            try:
                r = requests.post(f"{self.server_url}/embedding", json=test_payload, timeout=10)
                assert r.status_code == 200, f"Server not ready: {r.status_code}"
                self._log("✅ Server is ready!")
                break
            except (requests.exceptions.RequestException, AssertionError):
                self._log(f"⏳ Waiting for server to start... ({elapsed:.1f}s elapsed)")
                time.sleep(check_interval)

    def shutdown_server(self) -> None:
        """Shutdown the llama-server process"""
        if self.server_process:
            self._log("Shutting down server...")
            self.server_process.send_signal(signal.SIGINT)
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._log("Server did not shut down in time; killing process.")
                self.server_process.kill()
            self.server_process = None

    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown_server()

    def _image_to_data_url(self, image: Union[str, Image.Image]) -> str:
        """Convert image (path or PIL Image) to data URL"""
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Image must be str (file path) or PIL.Image, got {type(image)}")

        # Convert PIL Image to base64
        buffer = io.BytesIO()
        
        # Determine format - default to JPEG for compatibility
        format_map = {
            'JPEG': 'image/jpeg',
            'PNG': 'image/png',
            'WEBP': 'image/webp'
        }
        
        # Use original format if available, otherwise default to JPEG
        if hasattr(pil_image, 'format') and pil_image.format in format_map:
            save_format = pil_image.format
            mime_type = format_map[save_format]
        else:
            save_format = 'JPEG'
            mime_type = 'image/jpeg'
            # Convert to RGB if necessary for JPEG
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                pil_image = pil_image.convert('RGB')

        pil_image.save(buffer, format=save_format)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:{mime_type};base64,{image_data}"

    def _trim_text_with_tokenizer(self, text: str) -> str:
        """Trim text to max_text_length using the configured tokenizer"""
        if self.hf_tokenizer is None:
            return text
            
        tokens = self.hf_tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_text_length:
            tokens = tokens[:self.max_text_length]
        return self.hf_tokenizer.decode(tokens, clean_up_tokenization_spaces=True)

    def _process_content(self, content: str) -> str:
        """Process content with optional tokenizer trimming"""
        if self.hf_tokenizer is not None:
            return self._trim_text_with_tokenizer(content)
        return content

    def encode(self, items: List[EmbeddingRequestItem]) -> np.ndarray:
        """
        Encode items. Each item should be an EmbeddingRequestItem.
        """
        embeddings = []

        for i, item in tqdm(enumerate(items), total=len(items), desc="Encoding", unit="item"):
            # Process content with optional tokenizer trimming
            processed_content = self._process_content(item["content"])
            payload = {"content": processed_content}
            
            # Process image if present
            if item["image"] is not None:
                data_url = self._image_to_data_url(item["image"])
                payload["image"] = data_url
                
            is_image_request = item["image"] is not None
            response = requests.post(f"{self.server_url}/embedding", json=payload)
            assert response.status_code == 200, f"Server error: {response.text}"
            embedding_data = response.json()
            raw_embedding = embedding_data["embedding"]

            self._log(f"\n==========================")
            self._log(f"🧠 Item {i + 1} embedding response")
            self._log(f"📦 Type: {type(embedding_data).__name__}")
            self._log(f"🔑 Keys: {list(embedding_data.keys())}")
            self._log(f"🔎 Preview: {repr(embedding_data)[:500]}")
            self._log(f"🔍 Raw embedding type: {type(raw_embedding)}")
            self._log(f"🔍 Raw embedding shape: {np.array(raw_embedding).shape}")
            if self.hf_tokenizer and len(processed_content) != len(item["content"]):
                self._log(f"✂️ Text trimmed: {len(item['content'])} -> {len(processed_content)} chars")
            
            # Check if embeddings are already normalized
            embedding_array = np.array(raw_embedding)
            norms = np.linalg.norm(embedding_array, axis=1)
            if np.allclose(norms, 1.0, atol=1e-6):
                self._log(f"⚠️ WARNING: Raw embeddings appear to be already normalized!")
            
            # Handle image token extraction
            if is_image_request:
                start_idx = embedding_data["start_image_token_idx"]
                end_idx = embedding_data["end_image_token_idx"]    
                hidden_states = np.array(raw_embedding)
                # we need to capture <|vision_start|> ... <|vision_end|>
                image_embeddings = hidden_states[start_idx-1:end_idx+2]  
                pooled = image_embeddings.mean(axis=0)
                self._log(f"🖼️ Image token indices: start={start_idx}, end={end_idx}")
                self._log(f"🖼️ Extracted image embeddings shape: {image_embeddings.shape}")
                self._log(f"🖼️ Original total embeddings: {len(raw_embedding)}")
                self._log(f"🖼️ Image embeddings extracted: {len(image_embeddings)}")
            else:
                # Regular text processing - always mean pool the tokens
                hidden_states = np.array(raw_embedding)
                pooled = hidden_states.mean(axis=0)

            # Optional normalization
            if self.normalize:
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm
                    self._log(f"🔄 Applied L2 normalization")

            self._log(f"==========================")
            
            embeddings.append(pooled)

        return np.array(embeddings)