import requests # type: ignore
import os
import base64
import click # type: ignore
import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore


def clip_text(text, max_len=10):
    """Clip text to max_len characters, showing first part + '...' if needed"""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


class JinaEmbeddingsModel:
    def __init__(self, api_key, model="jina-embeddings-v4"):
        self.api_key = api_key
        self.model = model
        self.url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

    def encode(self, sentences):
        # Group inputs by type and task
        query_inputs = []
        passage_inputs = []
        
        for sentence in sentences:
            if sentence.startswith('[QUERY] '):
                content = sentence[8:]  # Remove '[QUERY] '
                query_inputs.append({"text": content})
            elif sentence.startswith('[DOCUMENT] '):
                content = sentence[11:]  # Remove '[DOCUMENT] '
                passage_inputs.append({"text": content})
            elif sentence.startswith('[IMAGE] '):
                image_path = sentence[8:]  # Remove '[IMAGE] '
                # Read and encode image as base64
                try:
                    with open(image_path, 'rb') as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    passage_inputs.append({"image": image_data})
                    print(f"🖼️ Loaded image: {image_path}")
                except FileNotFoundError:
                    print(f"❌ Image not found: {image_path}")
                    continue
            else:
                # Default to passage for unlabeled content
                passage_inputs.append({"text": sentence})
        
        embeddings = []
        original_order = []  # Track original order for reconstruction
        
        # Process queries
        if query_inputs:
            print(f"📤 Sending {len(query_inputs)} queries to Jina API...")
            query_data = {
                "model": self.model,
                "task": "retrieval.query",
                "input": query_inputs
            }
            response = requests.post(self.url, json=query_data, headers=self.headers)
            if response.status_code == 200:
                result = response.json()
                for item in result['data']:
                    embeddings.append(np.array(item['embedding']))
                    original_order.append(('query', len(original_order)))
            else:
                print(f"❌ Query API error: {response.status_code} - {response.text}")
                return None
        
        # Process passages (documents + images)
        if passage_inputs:
            print(f"📤 Sending {len(passage_inputs)} passages to Jina API...")
            passage_data = {
                "model": self.model,
                "task": "retrieval.passage",
                "input": passage_inputs
            }
            response = requests.post(self.url, json=passage_data, headers=self.headers)
            if response.status_code == 200:
                result = response.json()
                for item in result['data']:
                    embeddings.append(np.array(item['embedding']))
                    original_order.append(('passage', len(original_order)))
            else:
                print(f"❌ Passage API error: {response.status_code} - {response.text}")
                return None
        
        # Reconstruct original order
        ordered_embeddings = [None] * len(sentences)
        query_idx = 0
        passage_idx = 0
        
        for i, sentence in enumerate(sentences):
            if sentence.startswith('[QUERY] '):
                ordered_embeddings[i] = embeddings[query_idx]
                query_idx += 1
            else:
                ordered_embeddings[i] = embeddings[len(query_inputs) + passage_idx]
                passage_idx += 1
        
        return np.array(ordered_embeddings)

@click.command()
@click.option('--api-key', required=True, help='Jina API key')
@click.option('--model', default='jina-embeddings-v4', help='Jina model name')
@click.option('--input', 'input_path', required=True, help='Path to input txt file. Format: "[TYPE] content" where TYPE is QUERY, DOCUMENT, or IMAGE.')
@click.option('--output', 'output_path', required=True, help='Path to output md file for cosine similarity matrix.')
def main(api_key, model, input_path, output_path):
    # Load input sentences
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    
    print(f"📁 Loaded {len(raw_lines)} sentences from {input_path}")
    
    # Initialize Jina client
    client = JinaEmbeddingsModel(api_key=api_key, model=model)
    
    # Get embeddings
    embeddings = client.encode(raw_lines)
    
    if embeddings is None:
        print("❌ Failed to get embeddings")
        return

    # Extract display names from original texts  
    display_names = []
    for line in raw_lines:
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
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Cosine Similarity Matrix\n\n")
            
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
        
        print(f"📊 Saved cosine similarity matrix to {output_path}")

if __name__ == '__main__':
    main() # type: ignore