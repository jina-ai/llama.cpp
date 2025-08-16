import base64
import io
import os
import signal
import subprocess
import time
from typing import List, Optional, Tuple, Union

import numpy as np # type: ignore
import requests # type: ignore
from PIL import Image # type: ignore
from typing_extensions import TypedDict # type: ignore
from tqdm import tqdm # type: ignore
from transformers import AutoProcessor, AutoTokenizer # type: ignore


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
        hf_model_name: Optional[str] = None,
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
        self.hf_model_name = hf_model_name
        self.max_text_length = max_text_length
        self.server_process = None
        self.hf_tokenizer = None
        
        # Set server URL
        self.server_url = f"http://{host}:{port}"
        
        # Load tokenizer if specified
        if self.hf_model_name is not None:
            self._log(f"Loading HuggingFace processor and tokenizer for: {self.hf_model_name}")
            self.hf_processor = AutoProcessor.from_pretrained(self.hf_model_name)
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, use_fast=True)

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
            '--ubatch-size', str(self.ubatch_size),
            '--no-warmup'
        ]
        
        self._log(f"Starting llama-server with: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(cmd, env=env)

    def _wait_for_server(self, max_wait_time: int = 600, check_interval: float = 2.0) -> None:
        """Poll /health until the model is loaded (200). 503 means 'still loading'."""
        health_url = f"{self.server_url.rstrip('/')}/health"
        self._log(f"Waiting for server via {health_url} ...")

        deadline = time.monotonic() + max_wait_time
        last_status = None          # track last status to avoid repeating the same log
        last_heartbeat = 0.0        # rate-limit logs
        HEARTBEAT_EVERY = 20.0      # seconds

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Server did not become ready within {max_wait_time} seconds")

            try:
                r = requests.get(health_url, timeout=min(10, max(1, remaining)))
                if r.status_code == 200:
                    self._log("✅ Server is ready! (/health returned 200)")
                    return

                status = "Loading model" if r.status_code == 503 else f"HTTP {r.status_code}"
            except requests.RequestException as e:
                status = f"network error: {e}"

            # log only on status change or periodic heartbeat
            now = time.monotonic()
            if status != last_status or (now - last_heartbeat) >= HEARTBEAT_EVERY:
                elapsed = int(max_wait_time - remaining)
                self._log(f"⏳ {status}... ({elapsed}s elapsed)")
                last_status = status
                last_heartbeat = now

            time.sleep(min(check_interval, max(0.1, deadline - time.monotonic())))

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

    def _image_to_pixel_values(self, text: str, image: Union[str, Image.Image]) -> Tuple[str, List[int]]:
        """
        Convert image (path or PIL.Image) + text into Qwen2.5-VL pixel_values (patch embeddings),
        serialize them to raw float32 binary, and return (base64 string, [nx, ny, embd]).
        """
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Image must be str (file path) or PIL.Image, got {type(image)}")

        if not hasattr(self, "hf_processor"):
            raise RuntimeError("hf_processor is not initialized. Load it in __init__.")

        # Processor generates patch embeddings
        inputs = self.hf_processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].detach().cpu().numpy().astype("float32")  # (1, num_patches, embd)
        num_patches, embd = pixel_values.shape[1], pixel_values.shape[2]

        # Get grid shape
        if "patch_grid" in inputs:
            nx, ny = inputs["patch_grid"][0]
        else:
            nx = int(round(num_patches ** 0.5))
            ny = num_patches // nx
            if nx * ny != num_patches:
                raise ValueError(f"Cannot infer patch grid from {num_patches} patches")

        # Convert to raw float32 binary
        buf = pixel_values.tobytes(order="C")
        b64_data = base64.b64encode(buf).decode("utf-8")

        return b64_data, [nx, ny, embd]

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
        embeddings = []

        for i, item in tqdm(enumerate(items), total=len(items), desc="Encoding", unit="item"):
            processed_content = self._process_content(item["content"])
            payload = {"content": processed_content}
            
            if item["image"] is not None:
                data_url = self._image_to_data_url(item["image"])
                b64_bin, shape = self._image_to_pixel_values(item["content"], item["image"])
                
                payload["image"] = data_url
                payload["prebuilt_image"] = b64_bin
                payload["prebuilt_image_shape"] = shape # type: ignore
            
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
            
            embedding_array = np.array(raw_embedding)
            norms = np.linalg.norm(embedding_array, axis=1)
            if np.allclose(norms, 1.0, atol=1e-6):
                self._log(f"⚠️ WARNING: Raw embeddings appear to be already normalized!")
            
            if is_image_request:
                start_idx = embedding_data["start_image_token_idx"]
                end_idx = embedding_data["end_image_token_idx"]    
                hidden_states = embedding_array
                image_embeddings = hidden_states[start_idx-1:end_idx+2]  
                pooled = image_embeddings.mean(axis=0)
                self._log(f"🖼️ Image token indices: start={start_idx}, end={end_idx}")
                self._log(f"🖼️ Extracted image embeddings shape: {image_embeddings.shape}")
                self._log(f"🖼️ Original total embeddings: {len(raw_embedding)}")
                self._log(f"🖼️ Image embeddings extracted: {len(image_embeddings)}")
            else:
                hidden_states = embedding_array
                pooled = hidden_states.mean(axis=0)

            if self.normalize:
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm
                    self._log(f"🔄 Applied L2 normalization")

            self._log(f"==========================")
            
            embeddings.append(pooled)

        return np.array(embeddings)
