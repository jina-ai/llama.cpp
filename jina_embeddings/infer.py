import json
import os
import signal
import subprocess

import click # type: ignore
import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

from model import LlamaCppServerEmbeddingModel


def clip_text(text: str, max_len: int = 10) -> str:
    """Clip text to max_len characters, showing first part + '...' if needed"""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def save_cosine_similarity_matrix(raw_lines: list[str], embeddings: np.ndarray, save_path: str) -> None:
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
    
    print(f"Saved cosine similarity matrix to {save_path}")


@click.command()
@click.option('--llama-bin', default='./llama-server', help='Path to llama-server binary')
@click.option('--model', required=True, help='Path to model .gguf file')
@click.option('--mmproj', required=True, help='Path to mmproj .gguf file')
@click.option('--port', default=8080, help='Port for llama-server')
@click.option('--host', default='0.0.0.0', help='Host for llama-server')
@click.option('--ngl', default=999, help='Number of GPU layers')
@click.option('--gpus', default='0', help='CUDA_VISIBLE_DEVICES comma separated GPU ids (e.g. "0,1")')
@click.option('--input', 'input_path', required=True, help='Path to input txt file. Format: "[TYPE] content" where TYPE is QUERY, DOCUMENT, or IMAGE. For IMAGE, content should be the file path.')
@click.option('--output', 'output_path', required=True, help='Path to output JSON file for embeddings')
@click.option('--normalize-after-pooling', is_flag=True, default=False, help='Apply L2 normalization after pooling')
@click.option('--save-cosine-sim-path', help='Path to save cosine similarity matrix as markdown table')
@click.option('--query-prefix', default='Query: ', help='Prefix for [QUERY] lines')
@click.option('--document-prefix', default='Passage: ', help='Prefix for [DOCUMENT] lines')
@click.option('--image-prefix', default='Describe the image.<__image__>', help='Prefix for [IMAGE] lines')
def main(
    llama_bin, model, mmproj, port, host, ngl, gpus,
    input_path, output_path,
    normalize_after_pooling,
    save_cosine_sim_path, query_prefix, document_prefix, image_prefix
):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus

    cmd = [
        llama_bin,
        '-m', model,
        '--mmproj', mmproj,
        '--embedding',
        '--port', str(port),
        '-ngl', str(ngl),
        '--host', host,
        '--pooling', 'none'
    ]
    print(f"Starting llama-server with: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(raw_lines)} sentences from {input_path}")

        model = LlamaCppServerEmbeddingModel(
            server_url=f"http://{host}:{port}",
            normalize_after_pooling=normalize_after_pooling,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            image_prefix=image_prefix
        )

        model.wait_for_server()
        original_texts, embeddings = model.encode_from_lines(raw_lines)

        output_data = [
            {"text": text, "embedding": embedding.tolist()}
            for text, embedding in zip(original_texts, embeddings)
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(output_data, f_out, indent=2)

        print(f"Saved embeddings to {output_path}")

        # Save cosine similarity matrix if requested
        if save_cosine_sim_path:
            save_cosine_similarity_matrix(raw_lines, embeddings, save_cosine_sim_path)

    finally:
        print("Shutting down server...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Server did not shut down in time; killing process.")
            proc.kill()


if __name__ == '__main__':
    main() # type: ignore