import base64
import io
import os
import subprocess
import time
from typing import List, Optional, Tuple, Union

import numpy as np # type: ignore
import requests # type: ignore
from PIL import Image # type: ignore
from typing_extensions import TypedDict # type: ignore
from tqdm import tqdm # type: ignore
from transformers import Qwen2VLImageProcessorFast, Qwen2TokenizerFast # type: ignore


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
        ctx_size: int = 8192,
        ubatch_size: int = 8192,
        normalize: bool = False, 
        logging: bool = False,
        hf_model_name: Optional[str] = None,
        max_text_length: int = 512,
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

        if not hf_model_name:
            raise ValueError("hf_model_name must be provided to load the processor and tokenizer.")
        
        self.hf_image_processor = Qwen2VLImageProcessorFast.from_pretrained(self.hf_model_name)
        self.hf_tokenizer = Qwen2TokenizerFast.from_pretrained(self.hf_model_name)

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

    def _wait_for_server(self, max_wait_time: int = 600, interval: float = 2.0) -> None:
        url = f"{self.server_url.rstrip('/')}/health"
        deadline = time.monotonic() + max_wait_time
        while time.monotonic() < deadline:
            try:
                assert requests.get(url, timeout=5).status_code == 200
                return self._log("✅ Server is ready!")
            except Exception:
                self._log(f"⏳ Waiting for server via {url} ...")
            time.sleep(interval)

        raise TimeoutError(f"Server not ready within {max_wait_time}s")

    def shutdown_server(self) -> None:
        """Force kill the llama-server process"""
        if self.server_process:
            self._log("Killing server...")
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

    def _image_to_pixel_values(self, image: Union[str, Image.Image]) -> Tuple[str, List[int]]:
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

        # Processor generates patch embeddings
        inputs = self.hf_image_processor(images=[pil_image], return_tensors="pt")

        pixel_values = inputs["pixel_values"].detach().cpu().numpy().astype("float32")  # (1, num_patches, embd)
        image_grid_thw = inputs["image_grid_thw"].detach().cpu().numpy().tolist() 

        num_patches, embd = pixel_values.shape[0], pixel_values.shape[1]
        _, nx, ny = image_grid_thw[0]

        assert num_patches == nx * ny, f"Expected {nx}x{ny} patches, got {num_patches} patches"
    
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
                # NOTE: uncomment these two lines if you want to use normal processing pipeline 
                # data_url = self._image_to_data_url(item["image"])
                # payload["image"] = data_url
                b64_data, shape = self._image_to_pixel_values(item["image"])
                payload["prebuilt_image"] = b64_data
                payload["prebuilt_image_shape"] = shape # type: ignore
            
            is_image_request = item["image"] is not None
            response = requests.post(f"{self.server_url}/embedding", json=payload)
            assert response.status_code == 200, f"Server error: {response.text}"
            embedding_data = response.json()
            raw_embedding = embedding_data["embedding"]

            self._log(f"🧠 Item {i + 1} embedding response")
            self._log(f"🔍 Raw embedding shape: {np.array(raw_embedding).shape}")
            
            embedding_array = np.array(raw_embedding)
            
            if is_image_request:
                start_idx = embedding_data["start_image_token_idx"]
                end_idx = embedding_data["end_image_token_idx"]    
                image_embeddings = embedding_array[start_idx-1:end_idx+2]  
                pooled = image_embeddings.mean(axis=0)
                self._log(f"🖼️ Extracted image embeddings shape: {image_embeddings.shape}")
                self._log(f"🖼️ Image token indices: start={start_idx}, end={end_idx}")
                self._log(f"🖼️ Image embeddings extracted: {len(image_embeddings)}")
            else:
                pooled = embedding_array.mean(axis=0)

            if self.normalize:
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm
                    self._log(f"🔄 Applied L2 normalization")
            
            embeddings.append(pooled)

        return np.array(embeddings)
