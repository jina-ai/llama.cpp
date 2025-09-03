import os
import time
from typing import Optional

import click  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

from model import LlamaCppServerEmbeddingModel, EmbeddingRequestItem


@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--llama-bin", default="./llama-server", help="Path to llama-server binary")
@click.option("--model", required=True, help="Path to model .gguf file")
@click.option("--mmproj", required=True, help="Path to mmproj .gguf file")
@click.option("--port", default=8080, show_default=True, help="Port for llama-server")
@click.option("--host", default="0.0.0.0", show_default=True, help="Host for llama-server")
@click.option("-ngl", "--ngl", default=999, show_default=True, help="Number of GPU layers")
@click.option(
    "--gpus",
    default="0",
    show_default=True,
    help='CUDA_VISIBLE_DEVICES comma separated GPU ids (e.g. "0,1"). Empty string for CPU',
)
@click.option(
    "--hf-model-name",
    default="/Users/andrei/Downloads/jev4-retrieval",
    show_default=True,
    help="HuggingFace model name or path for tokenizer and image processor",
)
@click.option(
    "--max-text-length",
    default=512,
    show_default=True,
    help="Maximum text length in tokens (still used for any content string)",
)
@click.option(
    "--image-prefix",
    default="<|im_start|>user\n<__image__>Describe the image.<|im_end|>\n",
    show_default=False,
    help="Prefix prompt content used when sending the image",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    show_default=False,
    help="Exact file path to save the resulting embedding (.npy). If omitted, a random filename is generated in --output-base.",
)
@click.option(
    "--output-base",
    "output_base",
    default=".",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    show_default=True,
    help="Directory to save the .npy when --output is not provided.",
)
@click.option("--logging/--no-logging", default=True, show_default=True, help="Enable/disable verbose logs")
def main(
    image_path: str,
    llama_bin: str,
    model: str,
    mmproj: str,
    port: int,
    host: str,
    ngl: int,
    gpus: str,
    hf_model_name: str,
    max_text_length: int,
    image_prefix: str,
    output_path: Optional[str],
    output_base: str,
    logging: bool,
):
    """Embed a single image and save the vector to a .npy file.

    This starts llama-server if not already running via LlamaCppServerEmbeddingModel,
    performs a single image embedding request, and saves the pooled vector.
    """

    # Basic check and preview
    try:
        with Image.open(image_path) as im:
            im.verify()  # quick sanity check
    except Exception as e:
        raise click.ClickException(f"Failed to open image '{image_path}': {e}")

    # Create model with normalize=False and pool=False as requested
    model_client = LlamaCppServerEmbeddingModel(
        llama_bin=llama_bin,
        model_path=model,
        mmproj_path=mmproj,
        port=port,
        host=host,
        ngl=ngl,
        gpus=gpus,
        pool=False,
        normalize=False,
        logging=logging,
        hf_model_name=hf_model_name,
        max_text_length=max_text_length,
    )

    # Prepare single-item request
    item: EmbeddingRequestItem = EmbeddingRequestItem(
        content=image_prefix,
        image=image_path,
    )

    embeddings = model_client.encode([item])
    if embeddings.shape[0] == 1:
        embeddings = embeddings[0]

    # Determine output path: explicit --output takes precedence; otherwise generate random name under --output-base
    if output_path is None or len(str(output_path).strip()) == 0:
        os.makedirs(output_base, exist_ok=True)
        img_stem = os.path.splitext(os.path.basename(image_path))[0]
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        filename = f"{img_stem}-{ts}.npy"
        final_path = os.path.join(output_base, filename)
    else:
        # Ensure parent dir exists
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        final_path = output_path

    # Save to .npy
    np.save(final_path, embeddings)
    click.echo(
        f"Saved embedding with shape {embeddings.shape} to {os.path.abspath(final_path)}"
    )


if __name__ == "__main__":
    main()  # type: ignore
