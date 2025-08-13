#!/usr/bin/env python3
import os
import sys
import numpy as np # type: ignore
import torch # type: ignore

# transformers>=4.42 for Qwen2.5-VL; adjust if needed
from transformers import Qwen2_5_VLForConditionalGeneration # type: ignore

def save_bin_f32(t: torch.Tensor, path: str, label: str):
    t = t.float().contiguous().cpu().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"💾 Saving {label}: shape={t.shape} -> {path}")
    print(f"   First 10 vals: {t.flatten()[:10]}")
    t.tofile(path)
    print(f"✅ Wrote {t.size} float32 values")

def main():
    # Usage:
    #   python export_qwen25vl_patch_embed_bins.py /path/to/model_or_hub_id /out/dir
    # If args not provided, falls back to envs or defaults.
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QWEN25VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    out_dir    = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("QWEN_EXPORT_BIN_DIR", "./qwen25vl_bins")

    print("==============================================================")
    print(" Export Qwen2.5-VL patch_embed Conv3D weights -> flat & slices")
    print("==============================================================")
    print(f"Model: {model_name}")
    print(f"Out  : {out_dir}")

    # CPU only, no inference
    print("\n[1/3] Loading model on CPU…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,   # load as fp32 to avoid surprises
        device_map=None
    )

    # Find the Conv3D weight: visual.patch_embed.proj.weight
    # Expected shape: [embed_dim, in_channels=3, kT=2, kH=14, kW=14]
    print("\n[2/3] Locating patch_embed Conv3D weight…")
    state = model.state_dict()
    target_key = None
    for k in state.keys():
        # typical HF naming for this model family:
        # 'visual.patch_embed.proj.weight'
        if k.endswith("visual.patch_embed.proj.weight") or k == "visual.patch_embed.proj.weight":
            target_key = k
            break
        # fallback: print once if you can’t find it
    if target_key is None:
        # Try a more permissive search to help you debug names
        for k in state.keys():
            if "patch_embed" in k and k.endswith("weight"):
                print(f"Found candidate: {k} shape={tuple(state[k].shape)}")
        raise RuntimeError("Could not find 'visual.patch_embed.proj.weight' in model.state_dict().")

    W = state[target_key]  # torch.Tensor
    if W.ndim != 5:
        raise RuntimeError(f"Unexpected weight dim for {target_key}: {W.shape} (want [out, in, kT, kH, kW])")

    embed_dim, in_ch, kT, kH, kW = W.shape
    print(f"Found {target_key} with shape: {tuple(W.shape)}")
    assert in_ch == 3, f"in_channels expected 3, got {in_ch}"
    assert kT == 2, "temporal_patch_size must be 2 for this export"
    assert kH == 14 and kW == 14, f"patch_size must be 14, got {kH}x{kW}"

    # Slice along temporal dimension
    W_t0 = W[:, :, 0, :, :]  # [embed_dim, 3, 14, 14]
    W_t1 = W[:, :, 1, :, :]  # [embed_dim, 3, 14, 14]

    # Flatten to [embed_dim, 3*2*14*14] = [embed_dim, 1176]
    W_flat = W.contiguous().view(embed_dim, -1)

    print("\n[3/3] Saving bins (float32)…")
    save_bin_f32(W_flat, os.path.join(out_dir, "q25vl_patch_flat.bin"), "W_flat  [embed_dim, 1176]")
    save_bin_f32(W_t0,   os.path.join(out_dir, "q25vl_patch_t0.bin"),   "W_t0    [embed_dim, 3, 14, 14]")
    save_bin_f32(W_t1,   os.path.join(out_dir, "q25vl_patch_t1.bin"),   "W_t1    [embed_dim, 3, 14, 14]")

    print("\n🎉 Done. Use your GGUF patcher to overwrite:")
    print("  - v.patch_embd.weight_flat  <- q25vl_patch_flat.bin")
    print("  - v.patch_embd.weight       <- q25vl_patch_t0.bin")
    print("  - v.patch_embd.weight.1     <- q25vl_patch_t1.bin")

if __name__ == "__main__":
    main()
