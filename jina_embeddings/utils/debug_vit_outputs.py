import sys
import os

# Ensure vit_debugger.py is imported before anything else
sys.path.append(os.path.join(os.path.dirname(__file__)))
from vit_debugger import vit_debugger, LogParams

import click  # type: ignore
import torch  # type: ignore

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor  # type: ignore

from qwen_vl_utils import process_vision_info  # type: ignore


def setup_hooks(model, max_layers: int = 32):
    """Set up hooks to capture intermediate tensors."""
    vision_model = getattr(model, 'visual', None)
    if vision_model is None or not hasattr(vision_model, 'blocks'):
        raise ValueError("Model does not have a 'visual' attribute with 'blocks'. Please check the model structure.")

    # Patch embedding hook
    if hasattr(vision_model, 'patch_embed'):
        def patch_embed_hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                vit_debugger.capture_and_store("input_raw", input[0], log=False)
            vit_debugger.capture_and_store("patch_embeddings_final", output, log=False)
            return output
        vision_model.patch_embed.register_forward_hook(patch_embed_hook)
        print("✅ Added patch embedding hook")

    # Input to layers hook
    def input_to_layers_hook(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            vit_debugger.capture_and_store("input_to_layers", input[0], log=False)
        return output
    vision_model.blocks[0].register_forward_hook(input_to_layers_hook)
    print("✅ Added input_to_layers hook")

    # Transformer block hooks
    for i in range(min(max_layers, len(vision_model.blocks))):
        block = vision_model.blocks[i]
        if hasattr(block, 'norm1'):
            block.norm1.register_forward_hook(lambda m, inp, out, idx=i: vit_debugger.capture_and_store(f"norm1_{idx}", out, log=False))
        if hasattr(block, 'attn'):
            block.attn.register_forward_hook(lambda m, inp, out, idx=i: vit_debugger.capture_and_store(f"attn_out_{idx}", out, log=False))
        if hasattr(block, 'norm2'):
            block.norm2.register_forward_hook(lambda m, inp, out, idx=i: vit_debugger.capture_and_store(f"norm2_{idx}", out, log=False))
        if hasattr(block, 'mlp'):
            block.mlp.register_forward_hook(lambda m, inp, out, idx=i: vit_debugger.capture_and_store(f"ffn_out_{idx}", out, log=False))
        block.register_forward_hook(lambda m, inp, out, idx=i: vit_debugger.capture_and_store(f"layer_out_{idx}", out, log=False))

    # Merger hook
    if hasattr(vision_model, 'merger'):
        def merger_hook(module, input, output):
            vit_debugger.capture_and_store("merger_output", output, log=False)
            return output
        vision_model.merger.register_forward_hook(merger_hook)
        print("✅ Added merger hook")

    # Final ViT output
    def final_vit_hook(module, input, output):
        vit_debugger.capture_and_store("vit_final_embeddings", output, log=False)
        return output
    
    vision_model.register_forward_hook(final_vit_hook)
    print("✅ Added final ViT embeddings hook")


@click.command()
@click.option("--model-name", required=True, type=str, help="Path to the model directory.")
@click.option("--image-path", required=True, type=str, help="Path to the input image.")
@click.option("--output-file", required=True, type=str, help="Path to save the ViT debugger outputs.")
@click.option("--max-layers", default=32, type=int, show_default=True, help="Maximum number of layers to hook.")
def main(model_name, image_path, output_file, max_layers):
    # Configure debugger
    log_params = LogParams(start_patch=0, num_patches=5, start_head=0, num_heads=5, start_dim=0, num_dims=10)
    vit_debugger.log_file = output_file
    vit_debugger.params = log_params

    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(model_name)

    print("Setting up hooks...")
    setup_hooks(model, max_layers)

    print("Processing image...")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe the image."},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    print("Running inference...")
    with torch.no_grad():
        model(**inputs)
    print("✅ Inference completed")

    print("Saving outputs...")
    vit_debugger.save_outputs()
    print(f"✅ Outputs saved to {output_file}")


if __name__ == "__main__":
    main() # type: ignore
