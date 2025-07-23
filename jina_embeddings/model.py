import base64
import os
from typing import List, Optional, Tuple

import numpy as np # type: ignore
import requests # type: ignore
from typing_extensions import TypedDict # type: ignore


class EmbeddingRequestItem(TypedDict):
    content: str
    image: Optional[str]


class LlamaCppServerEmbeddingModel:
    def __init__(
        self, 
        server_url: str = "http://localhost:8080", 
        normalize_after_pooling: bool = False, 
        query_prefix: str = "Query: ", 
        document_prefix: str = "Passage: ", 
        image_prefix: str = "Describe the image.<__image__>"
    ) -> None:
        self.server_url = server_url
        self.normalize_after_pooling = normalize_after_pooling
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.image_prefix = image_prefix

    def _parse_line(self, line: str) -> Tuple[str, EmbeddingRequestItem]:
        """Parse input line and return (original_content, EmbeddingRequestItem)"""
        if line.startswith('[QUERY] '):
            content = line[8:]  # Remove '[QUERY] '
            item: EmbeddingRequestItem = { "content": self.query_prefix + content, "image": None }
            return content, item
        elif line.startswith('[DOCUMENT] '):
            content = line[11:]  # Remove '[DOCUMENT] '
            item: EmbeddingRequestItem = { "content": self.document_prefix + content, "image": None }
            return content, item
        elif line.startswith('[IMAGE] '):
            image_path = line[8:]  # Remove '[IMAGE] '
            data_url, success = self._process_image(image_path)
            assert success, f"Failed to process image: {image_path}"
            item: EmbeddingRequestItem = { "content": self.image_prefix, "image": data_url }
            return image_path, item
        else:
            raise ValueError(f"Invalid line format: {line}. Expected '[QUERY] ', '[DOCUMENT] ', or '[IMAGE] ' prefix.")

    def _process_image(self, image_path: str) -> Tuple[Optional[str], bool]:
        """Process image file and return (data_url, success)"""
        try:
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Detect image format from extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif ext == '.png':
                mime_type = 'image/png'
            elif ext == '.webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'  # default
            
            data_url = f"data:{mime_type};base64,{image_data}"
            return data_url, True
            
        except FileNotFoundError:
            print(f"❌ Image not found: {image_path}, processing as text only")
            return None, False

    def encode(self, items: List[EmbeddingRequestItem]) -> np.ndarray:
        """
        Encode items. Each item should be an EmbeddingRequestItem.
        """
        embeddings = []

        for i, item in enumerate(items):
            payload = {"content": item["content"], "image": item["image"]}
            is_image_request = item["image"] is not None
            response = requests.post(f"{self.server_url}/embedding", json=payload)
            assert response.status_code == 200, f"Server error: {response.text}"
            embedding_data = response.json()

            print(f"\n==========================")
            print(f"🧠 Item {i + 1} embedding response")
            print(f"📦 Type: {type(embedding_data).__name__}")
            print(f"🔑 Keys: {list(embedding_data.keys())}")
            print(f"🔎 Preview: {repr(embedding_data)[:500]}")
            print(f"==========================")

            raw_embedding = embedding_data["embedding"]
            
            print(f"🔍 Raw embedding type: {type(raw_embedding)}")
            print(f"🔍 Raw embedding shape: {np.array(raw_embedding).shape}")
            
            # Check if embeddings are already normalized
            embedding_array = np.array(raw_embedding)
            norms = np.linalg.norm(embedding_array, axis=1)
            if np.allclose(norms, 1.0, atol=1e-6):
                print(f"⚠️ WARNING: Raw embeddings appear to be already normalized!")
            
            # Handle image token extraction
            if is_image_request:
                start_idx = embedding_data["start_image_token_idx"]
                end_idx = embedding_data["end_image_token_idx"]
                
                print(f"🖼️ Image token indices: start={start_idx}, end={end_idx}")
                
                # Token-level embeddings - extract only image tokens
                hidden_states = np.array(raw_embedding)
                image_embeddings = hidden_states[start_idx:end_idx+1]  # +1 for inclusive end
                
                print(f"🖼️ Extracted image embeddings shape: {image_embeddings.shape}")
                print(f"🖼️ Original total embeddings: {len(raw_embedding)}")
                print(f"🖼️ Image embeddings extracted: {len(image_embeddings)}")
                
                # Pool only the image embeddings (always mean pool)
                pooled = image_embeddings.mean(axis=0)
                print(f"🖼️ Using mean pooling of image tokens")
                    
            else:
                # Regular text processing - always mean pool the tokens
                hidden_states = np.array(raw_embedding)
                pooled = hidden_states.mean(axis=0)
                print(f"📊 Applied mean pooling")

            # Optional normalization
            if self.normalize_after_pooling:
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm
                    print(f"🔄 Applied L2 normalization")

            embeddings.append(pooled)

        return np.array(embeddings)

    def encode_from_lines(self, raw_lines: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Process raw lines with type prefixes and return embeddings along with original content
        Returns: (original_texts, embeddings)
        """
        original_texts = []
        items = []
        
        for line in raw_lines:
            original, item = self._parse_line(line.strip())
            original_texts.append(original)
            items.append(item)
        
        embeddings = self.encode(items)
        return original_texts, embeddings