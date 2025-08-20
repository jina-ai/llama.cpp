import os
from typing import List, Tuple

import click # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

from model import LlamaCppServerEmbeddingModel, EmbeddingRequestItem


def clip_text(text: str, max_len: int = 10) -> str:
    """Clip text to max_len characters, showing first part + '...' if needed"""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def parse_line(
    line: str, 
    query_prefix: str = "Query: ", 
    document_prefix: str = "Passage: ", 
    image_prefix: str = "<__image__>"
) -> Tuple[str, EmbeddingRequestItem]:
    """Parse input line and return (original_content, EmbeddingRequestItem)"""

    if line.startswith('[QUERY] '):
        content = line[8:]  # Remove '[QUERY] '
        return content, EmbeddingRequestItem(
            content=query_prefix + content, 
            image=None
        )
    
    elif line.startswith('[DOCUMENT] '):
        content = line[11:]  # Remove '[DOCUMENT] '
        return content, EmbeddingRequestItem(
            content=document_prefix + content, 
            image=None
        )
    
    elif line.startswith('[IMAGE] '):
        image_path = line[8:]  # Remove '[IMAGE] '
        pil_image = Image.open(image_path)
        return image_path, EmbeddingRequestItem(
            content=image_prefix,
            image=pil_image
        )
    
    raise ValueError(f"Invalid line format: {line}. Expected '[QUERY] ', '[DOCUMENT] ', or '[IMAGE] ' prefix.")


def save_cosine_similarity_matrix(raw_lines: List[str], embeddings: np.ndarray, save_path: str) -> None:
    """Save cosine similarity matrix as markdown table"""
    # Extract display names from original texts  
    display_names = []
    for text in raw_lines:
        if text.startswith('[QUERY] '):
            content = text[8:]
            display_names.append(f"Q:{clip_text(content)}")
        elif text.startswith('[DOCUMENT] '):
            content = text[11:]
            display_names.append(f"D:{clip_text(content)}")
        elif text.startswith('[IMAGE] '):
            image_path = text[8:]
            filename = os.path.basename(image_path)
            display_names.append(f"I:{clip_text(filename)}")
        else:
            display_names.append(clip_text(text))
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create markdown table
    with open(save_path, 'w', encoding='utf-8') as f:
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


@click.command()
@click.option('--llama-bin', default='./llama-server', help='Path to llama-server binary')
@click.option('--model', required=True, help='Path to model .gguf file')
@click.option('--mmproj', required=True, help='Path to mmproj .gguf file')
@click.option('--port', default=8080, help='Port for llama-server')
@click.option('--host', default='0.0.0.0', help='Host for llama-server')
@click.option('--ngl', default=999, help='Number of GPU layers')
@click.option('--gpus', default='0', help='CUDA_VISIBLE_DEVICES comma separated GPU ids (e.g. "0,1")')
@click.option('--input', 'input_path', required=True, help='Path to input txt file. Format: "[TYPE] content" where TYPE is QUERY, DOCUMENT, or IMAGE. For IMAGE, content should be the file path.')
@click.option('--output', 'output_path', required=True, help='Path to output cosine similarity matrix as markdown table')
@click.option('--normalize', is_flag=True, default=False, help='Apply L2 normalization after pooling')
@click.option('--query-prefix', default='Query: ', help='Prefix for [QUERY] lines')
@click.option('--document-prefix', default='Passage: ', help='Prefix for [DOCUMENT] lines')
@click.option('--image-prefix', default='<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n', help='Prefix for [IMAGE] lines')
@click.option('--logging/--no-logging', default=True, help='Enable/disable logging output')
@click.option('--hf-model-name', default='/Users/andrei/Downloads/jev4-retrieval', help='HuggingFace model name for tokenizer and processor')
@click.option('--max-text-length', default=512, help='Maximum text length in tokens (requires --hf-tokenizer-name)')
def main(
    llama_bin, 
    model, 
    mmproj, 
    port, 
    host, 
    ngl, 
    gpus,
    input_path,
    output_path,
    normalize,
    query_prefix, 
    document_prefix, 
    image_prefix,
    logging, 
    hf_model_name, 
    max_text_length
):
    # Load input lines
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(raw_lines)} lines from {input_path}")

    # Create model - server starts automatically
    embedding_model = LlamaCppServerEmbeddingModel(
        llama_bin=llama_bin,
        model_path=model,
        mmproj_path=mmproj,
        port=port,
        host=host,
        ngl=ngl,
        gpus=gpus,
        normalize=normalize,
        logging=logging,
        hf_model_name=hf_model_name,
        max_text_length=max_text_length
    )

    # Parse lines and create embedding items
    items = []
    original_texts = []
    
    for line in raw_lines:
        try:
            original, item = parse_line(
                line.strip(),
                query_prefix=query_prefix,
                document_prefix=document_prefix,
                image_prefix=image_prefix
            )
            original_texts.append(original)
            items.append(item)
        except (ValueError, FileNotFoundError) as e:
            print(f"⚠️ Skipping line due to error: {e}")
            continue
    
    if not items:
        raise ValueError("No valid items to process after parsing lines")
    
    print(f"Successfully parsed {len(items)} items")

    # Generate embeddings
    embeddings = embedding_model.encode(items)
    print(f"Generated embeddings for {len(embeddings)} items")

    # Save cosine similarity matrix
    save_cosine_similarity_matrix(raw_lines, embeddings, output_path)
    print(f"Saved cosine similarity matrix to {output_path}")


if __name__ == '__main__':
    main() # type: ignore