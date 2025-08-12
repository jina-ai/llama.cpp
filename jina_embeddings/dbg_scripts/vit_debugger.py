# vit_debugger.py
import torch  # type: ignore
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np # type: ignore

@dataclass
class LogParams:
    start_patch: int = 0
    num_patches: int = 5
    start_head: int = 0
    num_heads: int = 5
    start_dim: int = 0
    num_dims: int = 10

class VitDebugger:
    def __init__(self, log_file: Optional[str] = None, params: Optional[LogParams] = None):
        self.log_file = log_file
        self.params = params if params else LogParams()
        self.layer_outputs: Dict[str, torch.Tensor] = {}

    # ---------- Tensor formatting ----------
    def _format_2d_tensor(self, tensor: torch.Tensor, name: str) -> str:
        rows, cols = tensor.shape
        lp = self.params
        start_patch = max(0, min(lp.start_patch, rows - 1))
        end_patch = min(start_patch + lp.num_patches, rows)
        start_dim = max(0, min(lp.start_dim, cols - 1))
        end_dim = min(start_dim + lp.num_dims, cols)

        lines = [f"=== {name} === Shape: [{rows}, {cols}]",
                 f"Logging patches {start_patch}-{end_patch-1}, dimensions {start_dim}-{end_dim-1}"]
        for patch in range(start_patch, end_patch):
            values = tensor[patch, start_dim:end_dim].float().tolist()
            values_str = " ".join(f"{v:.6f}" for v in values)
            lines.append(f"Patch {patch}: {values_str}")
            if end_dim < cols:
                lines[-1] += f" ... (dims {end_dim}-{cols-1})"
        return "\n".join(lines)

    def _format_3d_tensor(self, tensor: torch.Tensor, name: str) -> str:
        d0, d1, d2 = tensor.shape
        lines = [f"=== {name} === Shape: [{d0}, {d1}, {d2}]", "Logging first 3 slices:"]
        for i in range(min(3, d2)):
            lines.append(f"Slice {i}:")
            for r in range(min(3, d0)):
                vals = [f"{tensor[r, c, i].item():.6f}" for c in range(min(5, d1))]
                lines.append(f"  Row {r}: {' '.join(vals)}")
        return "\n".join(lines)

    def _format_4d_tensor(self, tensor: torch.Tensor, name: str) -> str:
        # tensor: [width, height, channels, batch]
        w, h, c, b = tensor.shape
        lp = self.params
        lines = [f"=== {name} === Shape: [{w}, {h}, {c}, {b}]",
                 f"Logging patches {lp.start_patch}-{lp.start_patch+lp.num_patches-1}, "
                 f"heads {lp.start_head}-{lp.start_head+lp.num_heads-1}, "
                 f"dimensions {lp.start_dim}-{lp.start_dim+lp.num_dims-1}"]
        for patch in range(lp.start_patch, min(lp.start_patch+lp.num_patches, c)):
            lines.append(f"Patch {patch}")
            for head in range(lp.start_head, min(lp.start_head+lp.num_heads, w)):
                row_vals = [f"{tensor[head, dim, patch, 0].item():.6f}"
                            for dim in range(lp.start_dim, min(lp.start_dim+lp.num_dims, h))]
                lines.append(f"  Head {head}: {' '.join(row_vals)} ... (dims {lp.start_dim+lp.num_dims}-{h-1})")
        return "\n".join(lines)

    # ---------- Logging ----------
    def log_tensor(self, tensor: torch.Tensor, name: str):
        if tensor.ndim == 2:
            msg = self._format_2d_tensor(tensor, name)
        elif tensor.ndim == 3:
            msg = self._format_3d_tensor(tensor, name)
        elif tensor.ndim == 4:
            msg = self._format_4d_tensor(tensor, name)
        else:
            raise ValueError(f"Unsupported tensor ndim: {tensor.ndim}")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")
        else:
            print(msg)

    def capture_and_store(self, name: str, tensor: torch.Tensor, log: bool = False):
        """Store tensor for later saving and optionally log immediately."""
        tensor_cpu = tensor.detach().cpu()
        self.layer_outputs[name] = tensor_cpu
        if log:
            self.log_tensor(tensor_cpu, name)

    def save_tensor_binary(self, tensor: torch.Tensor, filename: str):
        """Save any tensor as raw float32 binary."""
        tensor_f32 = tensor.detach().float().cpu().contiguous().numpy().astype(np.float32)
        print(f"💾 Saving tensor: shape {tensor_f32.shape} to {filename}")
        print(f"   First few values: {tensor_f32.flatten()[:10]}")
        tensor_f32.tofile(filename)
        print(f"✅ Saved {tensor_f32.size} float32 values to {filename}")

    # ---------- Output saving ----------
    def save_outputs(self):
        """Save all stored outputs in insertion order."""
        assert self.log_file, "Log file must be set to save outputs"
        with open(self.log_file, "w") as f:
            f.write("########## PYTORCH OUTPUTS ##########\n\n")
            for key, tensor in self.layer_outputs.items():
                if tensor is not None:
                    msg = self._format_2d_tensor(tensor, key) if tensor.ndim == 2 else \
                          self._format_3d_tensor(tensor, key) if tensor.ndim == 3 else \
                          self._format_4d_tensor(tensor, key) if tensor.ndim == 4 else \
                          f"=== {key} === Shape: {list(tensor.shape)} [Unsupported tensor dimension]"
                    f.write(msg + "\n\n")

# --- Singleton instance ---
log_params = LogParams(start_patch=0, num_patches=3, start_head=0, num_heads=5, start_dim=0, num_dims=10)
vit_debugger = VitDebugger(log_file="/home/andrei/workspace/jev4-vit-outputs.txt", params=log_params)
