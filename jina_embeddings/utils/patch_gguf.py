#!/usr/bin/env python3
import os
import numpy as np # type: ignore
import gguf

# Map GGUF dtype IDs to NumPy dtypes
GGUF_DTYPE_MAP = {
    0: np.float32,
    1: np.float16
}

def load_weights_from_bin(tensor, bin_path):
    """Load replacement weights from a .bin file, matching GGUF tensor dtype."""
    target_dtype = GGUF_DTYPE_MAP.get(tensor.tensor_type)
    if target_dtype is None:
        raise ValueError(f"❌ Unsupported tensor type {tensor.tensor_type} for {tensor.name}")

    file_size = os.path.getsize(bin_path)
    expected_size = tensor.n_bytes

    # Handle case where bin is float32 but GGUF expects float16
    if file_size == expected_size * 2 and target_dtype == np.float16:
        print(f"↪️ Converting {tensor.name} from float32 to float16")
        data = np.fromfile(bin_path, dtype=np.float32).astype(np.float16)
    else:
        data = np.fromfile(bin_path, dtype=target_dtype)

    if data.nbytes != expected_size:
        raise ValueError(f"❌ Size mismatch for {tensor.name}: got {data.nbytes}, expected {expected_size}")

    print(f"📦 Loaded {tensor.name}: shape {data.shape}, bytes {data.nbytes}")
    return data

def patch_tensor(output_gguf, tensor_name, bin_path):
    """Patch a single tensor in a GGUF file with data from a .bin file."""
    reader = gguf.GGUFReader(output_gguf)
    target_tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            target_tensor = t
            break

    if target_tensor is None:
        raise ValueError(f"❌ Tensor {tensor_name} not found in GGUF file")

    data = load_weights_from_bin(target_tensor, bin_path)

    with open(output_gguf, 'r+b') as f:
        f.seek(target_tensor.data_offset)
        f.write(data.tobytes())
        print(f"✅ Replaced {tensor_name} at offset {target_tensor.data_offset}")

def patch_multiple_tensors(input_gguf, output_gguf, patch_list):
    """
    Patch multiple tensors in one go.
    patch_list = [(tensor_name, bin_path), ...]
    """
    print(f"📄 Copying GGUF from {input_gguf} to {output_gguf}")
    with open(input_gguf, 'rb') as src, open(output_gguf, 'wb') as dst:
        dst.write(src.read())

    for tensor_name, bin_path in patch_list:
        patch_tensor(output_gguf, tensor_name, bin_path)

    print(f"🎉 Finished patching {len(patch_list)} tensors.")

if __name__ == "__main__":
    # Example usage
    input_gguf  = "/path/to/original.gguf"
    output_gguf = "/path/to/patched.gguf"

    patch_list = [
        ("v.patch_embd.weight_flat", "/path/to/weight_flat.bin"),
        ("v.patch_embd.weight.0",    "/path/to/conv2d_w0.bin"),
        ("v.patch_embd.weight.1",    "/path/to/conv2d_w1.bin")
    ]

    patch_multiple_tensors(input_gguf, output_gguf, patch_list)
