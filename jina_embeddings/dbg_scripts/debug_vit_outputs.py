import torch # type: ignore
from typing import Dict, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor # type: ignore
from qwen_vl_utils import process_vision_info # type: ignore
import numpy as np # type: ignore


class LogParams:
    """Parameters for controlling what gets logged"""
    def __init__(self, 
                 start_patch: int = 0,
                 num_patches: int = 5,
                 start_head: int = 0,
                 num_heads: int = 5,
                 start_dim: int = 0,
                 num_dims: int = 10):
        self.start_patch = start_patch
        self.num_patches = num_patches
        self.start_head = start_head
        self.num_heads = num_heads
        self.start_dim = start_dim
        self.num_dims = num_dims


class VitDebugger:
    """Enhanced debugger with parameterized logging capabilities"""
    
    def __init__(self, output_file: str = "vit_debug_output.txt", log_params: Optional[LogParams] = None):
        self.output_file = output_file
        self.layer_outputs: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self.log_params = log_params or LogParams()
    
    def create_hook(self, layer_name: str):
        """Create a forward hook that captures layer outputs"""
        def hook_fn(module, input, output):
            self.layer_outputs[layer_name] = output.detach().cpu()
            return output
        return hook_fn
    
    def save_embeddings_binary(self, tensor: torch.Tensor, filename: str):
        """Save embeddings as binary file for llama.cpp"""
        # Convert bfloat16 to float32 first, then to numpy
        tensor_f32 = tensor.float().cpu().numpy().astype(np.float32)
        print(f"💾 Saving embeddings: shape {tensor_f32.shape} to {filename}")
        print(f"   First few values: {tensor_f32.flatten()[:10]}")
        tensor_f32.tofile(filename)
        print(f"✅ Saved {tensor_f32.size} float32 values to {filename}")
    
    def clear_outputs(self):
        """Clear stored outputs"""
        self.layer_outputs.clear()
    
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _format_tensor(self, tensor: torch.Tensor, name: str) -> str:
        """Format tensor output matching llama.cpp style"""
        if len(tensor.shape) == 2:
            return self._format_2d_tensor(tensor, name)
        elif len(tensor.shape) == 3 or len(tensor.shape) == 4:
            return self._format_4d_tensor(tensor, name)
        else:
            return f"=== {name} === Shape: {list(tensor.shape)}\n[Unsupported tensor dimension]"
    
    def _format_2d_tensor(self, tensor: torch.Tensor, name: str) -> str:
        """Format 2D tensor output"""
        seq_len, features = tensor.shape
        start_patch = max(0, min(self.log_params.start_patch, seq_len - 1))
        end_patch = min(start_patch + self.log_params.num_patches, seq_len)
        start_dim = max(0, min(self.log_params.start_dim, features - 1))
        end_dim = min(start_dim + self.log_params.num_dims, features)
        
        lines = [f"=== {name} === Shape: [{seq_len}, {features}]"]
        lines.append(f"Logging patches {start_patch}-{end_patch-1}, dimensions {start_dim}-{end_dim-1}")
        
        for patch in range(start_patch, end_patch):
            values = tensor[patch, start_dim:end_dim].float().tolist()
            values_str = " ".join(f"{v:.6f}" for v in values)
            lines.append(f"Patch {patch}: {values_str}")
            if end_dim < features:
                lines[-1] += f" ... (dims {end_dim}-{features-1})"
        
        if end_patch < seq_len:
            lines.append(f"... (patches {end_patch}-{seq_len-1} not shown)")
        
        return "\n".join(lines)
    
    def _format_4d_tensor(self, tensor: torch.Tensor, name: str) -> str:
        """Format 3D tensor output (treating as 4D with batch=1)"""
        if len(tensor.shape) == 3:
            # Add batch dimension for consistent logging
            tensor = tensor.unsqueeze(-1)
        
        if len(tensor.shape) != 4:
            return f"=== {name} === Shape: {list(tensor.shape)} [Expected 3D or 4D tensor]"
        
        dim0, dim1, dim2, dim3 = tensor.shape
        start_patch = max(0, min(self.log_params.start_patch, dim0 - 1))
        end_patch = min(start_patch + self.log_params.num_patches, dim0)
        start_head = max(0, min(self.log_params.start_head, dim1 - 1))
        end_head = min(start_head + self.log_params.num_heads, dim1)
        start_dim = max(0, min(self.log_params.start_dim, dim2 - 1))
        end_dim = min(start_dim + self.log_params.num_dims, dim2)
        
        lines = [f"=== {name} === Shape: [{dim0}, {dim1}, {dim2}, {dim3}]"]
        lines.append(f"Logging patches {start_patch}-{end_patch-1}, heads {start_head}-{end_head-1}, dimensions {start_dim}-{end_dim-1}")
        
        for patch in range(start_patch, end_patch):
            lines.append(f"Patch {patch}")
            for head in range(start_head, end_head):
                values = tensor[patch, head, start_dim:end_dim, 0].float().tolist()
                values_str = " ".join(f"{v:.6f}" for v in values)
                lines.append(f"  Head {head}: {values_str}")
                if end_dim < dim2:
                    lines[-1] += f" ... (dims {end_dim}-{dim2-1})"
            if end_head < dim1:
                lines.append(f"  ... (heads {end_head}-{dim1-1} not shown)")
        
        if end_patch < dim0:
            lines.append(f"... (patches {end_patch}-{dim0-1} not shown)")
        
        return "\n".join(lines)
    
    def save_outputs(self, max_layers: int = 32):
        """Save captured outputs to file"""
        key_order = ['input_raw', 'patch_embeddings_final', 'input_to_layers']
        
        # Add layer outputs
        for i in range(max_layers):
            layer_keys = [f'norm1_{i}', f'attn_out_{i}', f'norm2_{i}', f'ffn_out_{i}', f'layer_out_{i}']
            key_order.extend(layer_keys)
        
        # Add final embeddings and merger
        key_order.extend(['merger_output', 'vit_final_embeddings'])
        
        with open(self.output_file, "w") as f:
            f.write("########## PYTORCH OUTPUTS ##########\n\n")
            for key in key_order:
                if key in self.layer_outputs:
                    output = self.layer_outputs[key]
                    formatted_output = self._format_tensor(output, key)
                    f.write(formatted_output + "\n\n")


def setup_hooks(model, debugger: VitDebugger, max_layers: int = 32):
    """Setup forward hooks using original Qwen processing style"""
    print(f"Setting up hooks for {max_layers} layers...")
    
    # Find vision model
    vision_model = None
    if hasattr(model, 'visual'):
        vision_model = model.visual
    elif hasattr(model, 'vision_model'):
        vision_model = model.vision_model
    elif hasattr(model, 'vision'):
        vision_model = model.vision
    else:
        print("ERROR: Cannot find vision model!")
        return
    
    if not hasattr(vision_model, 'blocks'):
        print(f"ERROR: Vision model doesn't have 'blocks' attribute.")
        return
    
    print(f"Found {len(vision_model.blocks)} vision blocks")

    # Hook patch embedding - using the same style as the old script
    if hasattr(vision_model, 'patch_embed'):
        def patch_embed_hook(module, input, output):
            # Capture raw input
            if isinstance(input, tuple) and len(input) > 0:
                debugger.layer_outputs["input_raw"] = input[0].detach().cpu()
                print(f"🔥 input_raw: {input[0].shape}")
            
            # Capture patch embeddings and save to binary
            output_cpu = output.clone().detach().cpu()
            debugger.layer_outputs["patch_embeddings_final"] = output_cpu
            print(f"🔥 patch_embeddings_final: {output.shape}")
            
            # Save to binary file for llama.cpp
            debugger.save_embeddings_binary(output_cpu, "/home/andrei/workspace/qwen25vl_patch_embeddings.bin")
            
            return output
        
        hook = vision_model.patch_embed.register_forward_hook(patch_embed_hook)
        debugger.hooks.append(hook)
        print("✅ Added patch embedding hook with binary save")

    # Hook input to first transformer block
    def input_to_layers_hook(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            debugger.layer_outputs["input_to_layers"] = input[0].detach().cpu()
            print(f"🔥 input_to_layers: {input[0].shape}")
        return output

    hook = vision_model.blocks[0].register_forward_hook(input_to_layers_hook)
    debugger.hooks.append(hook)
    print("✅ Added input_to_layers hook")

    # Setup hooks for transformer layers
    for i in range(min(max_layers, len(vision_model.blocks))):
        block = vision_model.blocks[i]
        
        # Hook components
        if hasattr(block, 'norm1'):
            hook = block.norm1.register_forward_hook(debugger.create_hook(f"norm1_{i}"))
            debugger.hooks.append(hook)
        
        if hasattr(block, 'norm2'):
            hook = block.norm2.register_forward_hook(debugger.create_hook(f"norm2_{i}"))
            debugger.hooks.append(hook)
        
        if hasattr(block, 'attn'):
            hook = block.attn.register_forward_hook(debugger.create_hook(f"attn_out_{i}"))
            debugger.hooks.append(hook)
        
        if hasattr(block, 'mlp'):
            hook = block.mlp.register_forward_hook(debugger.create_hook(f"ffn_out_{i}"))
            debugger.hooks.append(hook)
        
        hook = block.register_forward_hook(debugger.create_hook(f"layer_out_{i}"))
        debugger.hooks.append(hook)
    
    # Hook the patch merger - this is applied AFTER all transformer blocks
    if hasattr(vision_model, 'merger'):
        def merger_hook(module, input, output):
            merger_output = output.clone().detach().cpu()
            debugger.layer_outputs["merger_output"] = merger_output
            print(f"🔥 merger_output: {output.shape}")
            return output
        
        hook = vision_model.merger.register_forward_hook(merger_hook)
        debugger.hooks.append(hook)
        print("✅ Added merger hook")
    

    def final_vit_hook(module, input, output):
        # This captures the final output of the vision transformer
        # AFTER: patch_embed -> blocks -> merger -> reverse indexing
        final_embeddings = output.clone().detach().cpu()
        debugger.layer_outputs["vit_final_embeddings"] = final_embeddings
        print(f"🔥 🔥 🔥 TRUE FINAL ViT embeddings: {output.shape}")
        
        # Save the FINAL ViT embeddings to binary file
        debugger.save_embeddings_binary(final_embeddings, "/home/andrei/workspace/qwen25vl_final_vit_embeddings.bin")
        
        return output
    
    # Hook the entire vision model to capture its final output
    hook = vision_model.register_forward_hook(final_vit_hook)
    debugger.hooks.append(hook)
    print("✅ Added TRUE FINAL ViT embeddings hook (after merger + reordering)")
    print(f"✅ {len(debugger.hooks)} hooks registered")


def main():
    """Main function using original Qwen processing"""
    print("=" * 80)
    print("VIT DEBUGGER - USING ORIGINAL QWEN PROCESSING")
    print("=" * 80)
    
    # Configuration
    model_name = "/home/andrei/workspace/jev4-retrieval"
    image_path = "/home/andrei/workspace/dog.jpg"  # Use same image as old script
    output_file = "/home/andrei/workspace/qwen25_pytorch_vit_output.txt"
    max_layers = 32
    
    # Create log parameters
    log_params = LogParams(
        start_patch=0, num_patches=5, start_head=0, num_heads=5, start_dim=0, num_dims=10
    )
    
    # Create debugger
    debugger = VitDebugger(output_file, log_params)
    
    print("STEP 1: Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
    
    print("STEP 2: Setting up hooks...")
    setup_hooks(model, debugger, max_layers)
    
    print("STEP 3: Processing image using original Qwen method...")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe the image."},
        ],
    }]
    
    # Use the exact same processing as the old script
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    print("\n🖼️  IMAGE PREPROCESSING DEBUG:")
    if image_inputs:
        for i, img in enumerate(image_inputs):
            if hasattr(img, 'size'):
                print(f"  Image {i}: {img.size} (W x H)")
            elif hasattr(img, 'shape'):
                print(f"  Image {i}: shape {img.shape}")
            else:
                print(f"  Image {i}: {type(img)}")

    
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )

    print("\n🔍 PROCESSOR OUTPUT DEBUG:")
    if 'pixel_values' in inputs:
        print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    if 'image_grid_thw' in inputs:
        thw = inputs['image_grid_thw']
        print(f"  image_grid_thw: {thw}")
        if len(thw.shape) >= 2:
            for i in range(thw.shape[0]):
                t, h, w = thw[i].tolist()
                print(f"    Image {i}: T={t}, H={h}, W={w} -> Total patches: {t*h*w}")

    inputs = inputs.to("cuda")
    
    # Log input info
    print("Input information:")
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
    
    if 'image_grid_thw' in inputs:
        print(f"  grid_thw values: {inputs['image_grid_thw']}")
    
    print("\nSTEP 4: Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    print("✅ Inference completed")
    
    print("\nSTEP 5: Saving outputs...")
    debugger.save_outputs(max_layers=max_layers)
    
    print(f"✅ Outputs saved to {output_file}")
    print(f"📊 Captured {len(debugger.layer_outputs)} tensor outputs")
    print("✅ Patch embeddings saved to /home/andrei/workspace/qwen25vl_patch_embeddings.bin")
    print("🔥 🔥 🔥 TRUE FINAL ViT embeddings saved to /home/andrei/workspace/qwen25vl_final_vit_embeddings.bin")
    print("🎯 These embeddings include: patch_embed -> all_blocks -> merger -> reverse_indexing")
    
    debugger.cleanup_hooks()
    print("✅ Hooks cleaned up")


if __name__ == "__main__":
    main()