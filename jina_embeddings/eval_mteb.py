import logging
from pathlib import Path
from typing import List

import click  # type: ignore
import mteb  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore

from mteb.encoder_interface import PromptType  # type: ignore
from mteb.model_meta import ModelMeta  # type: ignore

from model import EmbeddingRequestItem, LlamaCppServerEmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Model metadata declaration
MODEL_META = ModelMeta(
    name="jina/jina-embeddings-v4",
    modalities=["text", "image"],
    languages=["eng-Latn"],
    similarity_fn_name="cosine",
    framework=["PyTorch"],  # Use valid framework from the allowed list
    open_weights=True,
    embed_dim=2048,
    max_tokens=512,
    revision="main",
    release_date="2024-01-01",
    n_parameters=3_000_000_000,  # Approximate for Jina v4
    memory_usage_mb=6000,
    license="apache-2.0",
    public_training_code="",
    public_training_data="",
    use_instructions=True,
    training_datasets={},  # Empty dict since we don't have training dataset info
)


class MTEBModelWrapper:
    """Wrapper to make LlamaCppServerEmbeddingModel compatible with MTEB"""
    
    mteb_model_meta = MODEL_META
    
    def __init__(
        self,
        embedding_model: LlamaCppServerEmbeddingModel,
        query_prefix: str = "Query: ",
        document_prefix: str = "Passage: ",
        image_prefix: str = '<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n',
        batch_size: int = 32,
    ):
        self.embedding_model = embedding_model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.image_prefix = image_prefix
        self.batch_size = batch_size

    def to_pil(self, image_data):
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        if isinstance(image_data, torch.Tensor):
            t = image_data
            # conver to HWC from CHW
            arr = t.permute(1, 2, 0).numpy()
            return Image.fromarray(arr).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image_data)}")

    def encode(
        self,
        sentences: List[str],
        task_name: str,
        prompt_type=None,
        **_,
    ) -> np.ndarray:
        """
        Encode text sentences for MTEB tasks
        """

        is_query = prompt_type == PromptType.query
        prefix = self.query_prefix if is_query else self.document_prefix
        logger.info(f"Encoding {len(sentences)} text inputs with prefix '{prefix}' for task {task_name}")

        # Apply prefix to sentences
        processed_sentences = [prefix + sent for sent in sentences]

        # Process in batches with progress bar
        all_embeddings = []
        
        with tqdm(total=len(sentences), desc=f"Encoding {task_name}", unit="sent") as pbar:
            for batch_idx in range(0, len(processed_sentences), self.batch_size):

                batch_sentences = processed_sentences[batch_idx:batch_idx + self.batch_size]

                batch_items = [
                    EmbeddingRequestItem(
                        content=sent, 
                        image=None
                    ) for sent in batch_sentences
                ]

                # Encode the batch using the embedding model
                batch_embeddings = self.embedding_model.encode(batch_items)
                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch_sentences))

        logger.info("Text encoding done.")
        return np.array(all_embeddings)

    def get_image_embeddings(
        self,
        images,
        **_,
    ) -> np.ndarray:
        """
        Encode images for MTEB image tasks
        """
        
        # NOTE: DataLoader yields batches, so we need to flatten them
        images_list = []
        for batch in images:
            if isinstance(batch, list):
                images_list.extend(batch)
            else:
                images_list.append(batch)
            
        logger.info(f"Encoding {len(images_list)} images")

        all_embeddings = []
        with tqdm(total=len(images_list), desc="Encoding images", unit="img") as pbar:
            for batch_idx in range(0, len(images_list), self.batch_size):
                batch_images = images_list[batch_idx:batch_idx + self.batch_size]
                
                batch_items = [
                    EmbeddingRequestItem(
                        content=self.image_prefix, 
                        image=self.to_pil(image)
                    ) for image in batch_images
                ]

                try:
                    batch_embeddings = self.embedding_model.encode(batch_items)
                    all_embeddings.extend(batch_embeddings)
                    pbar.update(len(batch_images))
                except Exception as e:
                    logger.error(f"Image batch encoding failed: {e}")
                    raise

        logger.info("Image encoding done.")
        return np.array(all_embeddings)

    # TODO: add prompt type handling for text embeddings here (for using the right prefix)
    def get_text_embeddings(
        self,
        texts: List[str],
        **_,
    ) -> np.ndarray:
        """
        Get text embeddings (same as encode but with query prefix by default)
        """
        logger.info(f"Encoding {len(texts)} text embeddings")
        
        # Apply document prefix to texts
        processed_texts = [self.query_prefix + text for text in texts]
        logger.info(f"First 100 characters of processed text: {processed_texts[0][:100]}")

        # Process in batches with progress bar
        all_embeddings = []
        
        with tqdm(total=len(texts), desc="Encoding text embeddings", unit="text") as pbar:
            for batch_idx in range(0, len(processed_texts), self.batch_size):
                batch_texts = processed_texts[batch_idx:batch_idx + self.batch_size]
                
                # Create EmbeddingRequestItems for the batch
                batch_items = []
                for text in batch_texts:
                    item: EmbeddingRequestItem = {
                        "content": text,
                        "image": None
                    }
                    batch_items.append(item)
                
                # Encode the batch using the embedding model
                try:
                    batch_embeddings = self.embedding_model.encode(batch_items)
                    all_embeddings.extend(batch_embeddings)
                    pbar.update(len(batch_texts))
                except Exception as e:
                    logger.error(f"Text batch encoding failed: {e}")
                    raise

        logger.info("Text embeddings done.")
        return np.array(all_embeddings)

@click.command()
@click.option('--llama-bin', required=True, help='Path to llama-server binary')
@click.option('--model', required=True, help='Path to model .gguf file')
@click.option('--mmproj', required=True, help='Path to mmproj .gguf file')
@click.option('--tasks', required=True, help='MTEB tasks to run (comma-separated, "vidore-v1", "vidore-v2", or "all")')
@click.option('--output-dir', required=True, help='Output directory for results')
@click.option('--port', default=8080, help='Port for llama-server')
@click.option('--host', default='127.0.0.1', help='Host for llama-server')
@click.option('--ngl', default=999, help='Number of GPU layers')
@click.option('--gpus', default='0', help='CUDA_VISIBLE_DEVICES')
@click.option('--normalize', is_flag=True, default=False, help='Apply L2 normalization after pooling')
@click.option('--ctx-size', default=4096, help='Context size for llama-server')
@click.option('--ubatch-size', default=4096, help='Physical maximum batch size for computation')
@click.option('--hf-pretrained', default="jinaai/jina-embeddings-v4", help='HuggingFace tokenizer model')
@click.option('--max-text-length', default=512, help='Maximum text length in tokens (requires HF tokenizer)')
@click.option('--query-prefix', default="Query: ", help='Prefix for query inputs')
@click.option('--document-prefix', default="Passage: ", help='Prefix for document inputs')
@click.option('--image-prefix', default='<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n', help='Prefix for image inputs')
@click.option('--batch-size', default=12, help="Number of sentences to process in each batch")
@click.option('--logging/--no-logging', default=True, help='Enable/disable model logging')
def main(
    llama_bin,
    model,
    mmproj,
    tasks,
    output_dir,
    port,
    host,
    ngl,
    gpus,
    normalize,
    ctx_size,
    ubatch_size,
    hf_pretrained,
    max_text_length,
    query_prefix,
    document_prefix,
    image_prefix,
    batch_size,
    logging
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load HuggingFace tokenizer if specified
    logger.info(f"Tokenizer and processor will be loaded by model: {hf_pretrained}")

    # Parse tasks - handle special cases for ViDoRe
    task_list = [t.strip() for t in tasks.split(",")]
    logger.info(f"Running specified tasks: {task_list}")

    # Create embedding model - server starts automatically
    embedding_model = LlamaCppServerEmbeddingModel(
        llama_bin=llama_bin,
        model_path=model,
        mmproj_path=mmproj,
        port=port,
        host=host,
        ngl=ngl,
        gpus=gpus,
        ctx_size=ctx_size,
        ubatch_size=ubatch_size,
        normalize=normalize,
        logging=logging,
        hf_model_name=hf_pretrained,
        max_text_length=max_text_length
    )

    # Create MTEB wrapper
    model_wrapper = MTEBModelWrapper(
        embedding_model=embedding_model,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        image_prefix=image_prefix,
        batch_size=batch_size,
    )

    logger.info("Starting MTEB evaluation...")
    evaluation = mteb.MTEB(tasks=task_list)
    evaluation.run(model_wrapper, output_folder=output_dir, overwrite_results=True)
    logger.info(f"MTEB evaluation complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main() # type: ignore