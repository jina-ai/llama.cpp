import json
import os
import click
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel
from PIL import Image
import requests
from io import BytesIO

class JinaEmbeddingsHFModel:
    def __init__(self, model_name="/home/andrei/workspace/jev4-hf", device="cuda"):
        """Initialize the HuggingFace Jina model"""
        print(f"🔄 Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        )
        
        # Set device
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ Model loaded on device: {self.device}")

    def _load_image(self, image_path):
        """Load image from local path or URL"""
        try:
            if image_path.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                # Load from local path
                image = Image.open(image_path)
            return image
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None

    def encode(self, sentences):
        """Encode sentences using the HuggingFace model"""
        # Separate different types of inputs
        query_texts = []
        passage_texts = []
        image_paths = []
        
        # Track original order for reconstruction
        input_mapping = []
        
        for i, sentence in enumerate(sentences):
            if sentence.startswith('[QUERY] '):
                content = sentence[8:]  # Remove '[QUERY] '
                query_texts.append(content)
                input_mapping.append(('query', len(query_texts) - 1))
            elif sentence.startswith('[DOCUMENT] '):
                content = sentence[11:]  # Remove '[DOCUMENT] '
                passage_texts.append(content)
                input_mapping.append(('passage', len(passage_texts) - 1))
            elif sentence.startswith('[IMAGE] '):
                image_path = sentence[8:]  # Remove '[IMAGE] '
                image_paths.append(image_path)
                input_mapping.append(('image', len(image_paths) - 1))
            else:
                # Default to passage for unlabeled content
                passage_texts.append(sentence)
                input_mapping.append(('passage', len(passage_texts) - 1))
        
        # Store embeddings for each type
        query_embeddings = []
        passage_embeddings = []
        image_embeddings = []
        
        # Encode queries
        if query_texts:
            print(f"🔍 Encoding {len(query_texts)} queries...")
            with torch.no_grad():
                query_embs = self.model.encode_text(
                    texts=query_texts,
                    task="retrieval",
                    prompt_name="query"
                )
                # Handle both tensor and list returns
                if isinstance(query_embs, torch.Tensor):
                    query_embeddings = query_embs.detach().cpu().numpy()
                elif isinstance(query_embs, list):
                    # Check if list contains tensors
                    if query_embs and isinstance(query_embs[0], torch.Tensor):
                        query_embeddings = np.array([emb.detach().cpu().numpy() for emb in query_embs])
                    else:
                        query_embeddings = np.array(query_embs)
                else:
                    query_embeddings = np.array(query_embs)
        
        # Encode passages
        if passage_texts:
            print(f"📄 Encoding {len(passage_texts)} passages...")
            with torch.no_grad():
                passage_embs = self.model.encode_text(
                    texts=passage_texts,
                    task="retrieval",
                    prompt_name="passage"
                )
                # Handle both tensor and list returns
                if isinstance(passage_embs, torch.Tensor):
                    passage_embeddings = passage_embs.detach().cpu().numpy()
                elif isinstance(passage_embs, list):
                    # Check if list contains tensors
                    if passage_embs and isinstance(passage_embs[0], torch.Tensor):
                        passage_embeddings = np.array([emb.detach().cpu().numpy() for emb in passage_embs])
                    else:
                        passage_embeddings = np.array(passage_embs)
                else:
                    passage_embeddings = np.array(passage_embs)
        
        # Encode images (no prompt_name parameter for images)
        if image_paths:
            print(f"🖼️ Encoding {len(image_paths)} images...")
            valid_images = []
            valid_indices = []
            
            for idx, image_path in enumerate(image_paths):
                image = self._load_image(image_path)
                if image is not None:
                    valid_images.append(image_path)
                    valid_indices.append(idx)
                    print(f"✅ Loaded image: {os.path.basename(image_path)}")
            
            if valid_images:
                with torch.no_grad():
                    img_embs = self.model.encode_image(
                        images=valid_images,
                        task="retrieval"
                    )
                    # Handle both tensor and list returns
                    if isinstance(img_embs, torch.Tensor):
                        img_embs_np = img_embs.detach().cpu().numpy()
                    elif isinstance(img_embs, list):
                        # Check if list contains tensors
                        if img_embs and isinstance(img_embs[0], torch.Tensor):
                            img_embs_np = np.array([emb.detach().cpu().numpy() for emb in img_embs])
                        else:
                            img_embs_np = np.array(img_embs)
                    else:
                        img_embs_np = np.array(img_embs)
                    
                    # Create full array with None for failed images
                    full_image_embeddings = [None] * len(image_paths)
                    
                    for i, valid_idx in enumerate(valid_indices):
                        full_image_embeddings[valid_idx] = img_embs_np[i]
                    
                    image_embeddings = full_image_embeddings
        
        # Reconstruct embeddings in original order
        ordered_embeddings = []
        
        for input_type, index in input_mapping:
            if input_type == 'query':
                ordered_embeddings.append(query_embeddings[index])
            elif input_type == 'passage':
                ordered_embeddings.append(passage_embeddings[index])
            elif input_type == 'image':
                if index < len(image_embeddings) and image_embeddings[index] is not None:
                    ordered_embeddings.append(image_embeddings[index])
                else:
                    # Skip failed images
                    print(f"⚠️ Skipping failed image at index {index}")
                    continue
        
        if not ordered_embeddings:
            print("❌ No valid embeddings generated")
            return None
        
        return np.array(ordered_embeddings)


@click.command()
@click.option('--model', default='/home/andrei/workspace/jev4-hf', help='HuggingFace model name')
@click.option('--device', default='cuda', help='Device to use (cuda/cpu)')
@click.option('--input', 'input_path', required=True, help='Path to input txt file. Format: "[TYPE] content" where TYPE is QUERY, DOCUMENT, or IMAGE.')
@click.option('--output', 'output_path', required=True, help='Path to output JSON file for embeddings')
@click.option('--save-cosine-sim-path', help='Path to save cosine similarity matrix as markdown table')
def main(model, device, input_path, output_path, save_cosine_sim_path):
    """
    Generate embeddings using HuggingFace Jina model for retrieval tasks.
    
    Input file format:
    [QUERY] your search query
    [DOCUMENT] your document text
    [IMAGE] path/to/image.jpg or https://example.com/image.jpg
    """
    
    # Load input sentences
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_path}")
        return
    
    print(f"📁 Loaded {len(raw_lines)} sentences from {input_path}")
    
    # Initialize HuggingFace client
    try:
        client = JinaEmbeddingsHFModel(model_name=model, device=device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get embeddings
    embeddings = client.encode(raw_lines)
    
    if embeddings is None:
        print("❌ Failed to get embeddings")
        return
    
    # Extract content for output (remove type prefixes)
    clean_texts = []
    valid_lines = []  # Track which lines produced valid embeddings
    
    embedding_idx = 0
    for line in raw_lines:
        # Check if this line would produce a valid embedding
        if line.startswith('[IMAGE] '):
            image_path = line[8:]
            # Try to load image to see if it's valid
            try:
                if image_path.startswith(('http://', 'https://')):
                    response = requests.get(image_path, timeout=10)
                    Image.open(BytesIO(response.content))
                else:
                    Image.open(image_path)
                # If we get here, image is valid
                clean_texts.append(image_path)
                valid_lines.append(line)
                embedding_idx += 1
            except:
                # Skip invalid images
                continue
        else:
            # Text inputs are always valid
            if line.startswith('[QUERY] '):
                clean_texts.append(line[8:])
            elif line.startswith('[DOCUMENT] '):
                clean_texts.append(line[11:])
            else:
                clean_texts.append(line)
            valid_lines.append(line)
            embedding_idx += 1
    
    # Save embeddings
    output_data = [
        {"text": text, "embedding": embedding.tolist()}
        for text, embedding in zip(clean_texts, embeddings)
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Saved {len(output_data)} embeddings to {output_path}")
    
    # Save cosine similarity matrix if requested
    if save_cosine_sim_path:
        def clip_text(text, max_len=10):
            """Clip text to max_len characters, showing first part + '...' if needed"""
            if len(text) <= max_len:
                return text
            return text[:max_len-3] + "..."
        
        # Extract display names from valid lines
        display_names = []
        for line in valid_lines:
            if line.startswith('[QUERY] '):
                content = line[8:]
                display_names.append(f"Q:{clip_text(content)}")
            elif line.startswith('[DOCUMENT] '):
                content = line[11:]
                display_names.append(f"D:{clip_text(content)}")
            elif line.startswith('[IMAGE] '):
                image_path = line[8:]
                filename = os.path.basename(image_path)
                display_names.append(f"I:{clip_text(filename)}")
            else:
                display_names.append(clip_text(line))
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create markdown table
        with open(save_cosine_sim_path, 'w', encoding='utf-8') as f:
            f.write("# Cosine Similarity Matrix\n\n")
            f.write(f"Generated using HuggingFace model: `{model}`\n\n")
            
            # Write header row
            f.write("| Item |")
            for name in display_names:
                f.write(f" {name} |")
            f.write("\n")
            
            # Write separator row
            f.write("|" + "---|" * (len(display_names) + 1) + "\n")
            
            # Write data rows
            for i, row_name in enumerate(display_names):
                f.write(f"| {row_name} |")
                for j in range(len(display_names)):
                    sim_score = similarity_matrix[i, j]
                    f.write(f" {sim_score:.3f} |")
                f.write("\n")
        
        print(f"📊 Saved cosine similarity matrix to {save_cosine_sim_path}")


if __name__ == '__main__':
    main()