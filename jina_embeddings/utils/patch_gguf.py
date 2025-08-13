import os
import numpy as np # type: ignore
import gguf

# Type mapping from gguf.tensor_type to numpy dtype
GGUF_DTYPE_MAP = {
    0: np.float32,  # assuming these codes correspond
    1: np.float16,
    # extend if needed...
}

def load_weights_for_tensor(bin_path: str, tensor) -> np.ndarray:
    target_dtype = GGUF_DTYPE_MAP.get(tensor.tensor_type)
    if target_dtype is None:
        raise ValueError(f"Unsupported tensor_type {tensor.tensor_type} for {tensor.name}")
    sz = os.path.getsize(bin_path)
    expected = tensor.n_bytes

    if target_dtype == np.float16 and sz == expected * 2:
        arr = np.fromfile(bin_path, dtype=np.float32).astype(np.float16)
    else:
        arr = np.fromfile(bin_path, dtype=target_dtype)

    if arr.nbytes != expected:
        raise ValueError(f"Size mismatch for {tensor.name}: {arr.nbytes} != {expected}")
    return arr

def patch_tensor(gguf_path: str, tensor, new_data: np.ndarray):
    with open(gguf_path, "r+b") as f:
        f.seek(tensor.data_offset)
        old_bytes = f.read(tensor.n_bytes)
        old_arr = np.frombuffer(old_bytes, dtype=new_data.dtype)

        diff = old_arr - new_data
        mae = np.abs(diff).mean()
        mse = (diff ** 2).mean()
        max_abs = np.abs(diff).max()
        mean_signed = diff.mean()

        print(f"🔍 {tensor.name}: MAE={mae}, MSE={mse}, MaxAbs={max_abs}, MeanSigned={mean_signed}")

        f.seek(tensor.data_offset)
        f.write(new_data.tobytes())

def patch_multiple(input_gguf: str, output_gguf: str, patch_list):
    print(f"Copying GGUF from {input_gguf} to {output_gguf}")
    with open(input_gguf, "rb") as src, open(output_gguf, "wb") as dst:
        dst.write(src.read())

    reader = gguf.GGUFReader(output_gguf)
    name_to_tensor = {t.name: t for t in reader.tensors}

    for name, bin_path in patch_list:
        if name not in name_to_tensor:
            raise ValueError(f"Tensor '{name}' not found")
        print(f"Patching {name}")
        t = name_to_tensor[name]
        new_data = load_weights_for_tensor(bin_path, t)
        patch_tensor(output_gguf, t, new_data)

    gguf.GGUFReader(output_gguf)  # quick validity check
    print("✅ Patching complete, GGUF loads successfully")

if __name__ == "__main__":
    # Example usage
    input_gguf  = "/Users/andrei/Documents/gguf/mmproj-jev4-bf16.gguf"
    output_gguf = "/Users/andrei/Documents/gguf/mmproj-jev4-bf16-fixed.gguf"

    patch_list = [
        ("v.patch_embd.weight_flat", "/Users/andrei/Documents/gguf/q25vl_patch_flat.bin"),
        # ("v.patch_embd.weight",    "/Users/andrei/Documents/gguf/q25vl_patch_t0.bin"),
        # ("v.patch_embd.weight.1",    "/Users/andrei/Documents/gguf/q25vl_patch_t1.bin")
    ]

    patch_multiple(input_gguf, output_gguf, patch_list)
