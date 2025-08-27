import os
import time
import uuid
from typing import Optional, Tuple

import click  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore
from transformers import Qwen2VLImageProcessorFast  # type: ignore
import matplotlib  # type: ignore


def _infer_grid_from_image(image_path: str, hf_model_name: str) -> Tuple[int, int]:
    """Return (nx, ny) patch grid for the given image using Qwen2VL processor.

    nx = number of patches along width (x)
    ny = number of patches along height (y)
    """
    processor = Qwen2VLImageProcessorFast.from_pretrained(hf_model_name, max_pixels=602112)
    with Image.open(image_path) as pil:
        inputs = processor(images=[pil], return_tensors="pt")
    # image_grid_thw: (batch, 3) -> [t, nx, ny]
    image_grid_thw = inputs["image_grid_thw"].detach().cpu().numpy().tolist()
    _, nx, ny = image_grid_thw[0]
    return int(nx), int(ny)


def _compute_diffs(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Compute per-row difference between a and b.

    a, b: shape (num_patches, dim)
    metric: 'cosine' | 'linf'
    returns: shape (num_patches,)
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert a.ndim == 2, f"Expected 2D arrays (num_patches, dim), got {a.ndim}D"

    if metric == "cosine":
        # 1 - cosine similarity
        eps = 1e-8
        a_norm = np.linalg.norm(a, axis=1, keepdims=True) + eps
        b_norm = np.linalg.norm(b, axis=1, keepdims=True) + eps
        sim = np.sum(a * b, axis=1, keepdims=False) / (a_norm.flatten() * b_norm.flatten())
        diffs = 1.0 - np.clip(sim, -1.0, 1.0)
    elif metric == "linf":
        diffs = np.max(np.abs(a - b), axis=1)
    else:
        raise click.ClickException(f"Unsupported metric: {metric}")

    return diffs.astype(np.float32)


def _make_overlay(
    heatmap_values: np.ndarray,   # normalized [0,1] for coloring
    raw_values: np.ndarray,       # raw metric values (cosine distance or L∞)
    nx: int,
    ny: int,
    img_w: int,
    img_h: int,
    alpha: float,
    topk: int = 10,               # annotate only the top-K highest-difference patches
) -> Image.Image:
    """Create a green→yellow→red RGBA heatmap overlay with patch borders then top-K raw values."""

    # --- safety checks ---
    assert heatmap_values.shape == raw_values.shape, "heatmap_values and raw_values must match"
    vals = np.clip(heatmap_values, 0.0, 1.0)

    # Arrange into 2D grids (ny rows, nx cols). reshape((nx, ny)).T -> (ny, nx)
    grid = vals.reshape((nx, ny)).T            # (ny, nx)
    grid_raw = raw_values.reshape((nx, ny)).T  # (ny, nx)

    # Blocky upsample to image size (no smoothing)
    grid_img = Image.fromarray((grid * 255).astype(np.uint8))
    grid_img = grid_img.resize((img_w, img_h), resample=Image.NEAREST)

    # Colormap (0=green, 1=red) with modern API
    cmap = matplotlib.colormaps.get_cmap("RdYlGn_r") # type: ignore
    rgba = (cmap(np.asarray(grid_img) / 255.0) * 255).astype(np.uint8)

    # Global alpha scaling
    rgba[..., 3] = (rgba[..., 3].astype(np.float32) * alpha).astype(np.uint8)

    overlay = Image.fromarray(rgba)
    draw = ImageDraw.Draw(overlay)

    # Patch geometry
    w_patch = img_w / nx
    h_patch = img_h / ny

    # ---- Patch borders (semi-transparent white) FIRST ----
    border_alpha = int(255 * 0.20)  # 20% opacity
    border_color = (255, 255, 255, border_alpha)
    for j in range(ny):
        y0 = int(round(j * h_patch))
        y1 = int(round((j + 1) * h_patch))
        for i in range(nx):
            x0 = int(round(i * w_patch))
            x1 = int(round((i + 1) * w_patch))
            draw.rectangle([x0, y0, x1, y1], outline=border_color, width=1)

    # ---- Top-K labels (by normalized value), showing RAW metric, drawn LAST ----
    if topk and topk > 0:
        try:
            font = ImageFont.truetype(
                "DejaVuSans-Bold.ttf",
                size=max(10, min(int(img_w // nx), int(img_h // ny)) // 2),
            )
        except Exception:
            font = ImageFont.load_default()

        flat_vals = grid.flatten()     # (ny*nx,)
        flat_raw = grid_raw.flatten()  # (ny*nx,)
        k = min(topk, flat_vals.size)
        top_idx = np.argpartition(flat_vals, -k)[-k:]
        # sort ascending so the largest gets drawn last (on very top)
        top_idx = top_idx[np.argsort(flat_vals[top_idx])]

        for idx in top_idx:
            j, i = divmod(idx, nx)  # grid is (ny, nx)
            raw_val = float(flat_raw[idx])
            cx = int((i + 0.5) * w_patch)
            cy = int((j + 0.5) * h_patch)
            draw.text(
                (cx, cy),
                f"{raw_val:.3f}",
                font=font,
                fill=(255, 255, 255, 255),
                anchor="mm",
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
            )

    return overlay


@click.command()
@click.argument("npy_a", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("npy_b", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--hf-model-name", required=True,
              help="HuggingFace model name/path for Qwen2-VL image processor")
@click.option("--metric", type=click.Choice(["cosine", "linf"], case_sensitive=False),
              default="cosine", show_default=True,
              help="Difference metric per patch: cosine (1 - cos sim) or linf (max abs diff)")
@click.option("--alpha", default=0.6, show_default=True,
              help="Overlay alpha scaling factor (0..1)")
@click.option("--percentile", default=99.0, show_default=True,
              help="Clip heatmap values to this percentile before normalization")
@click.option("--output", "output_path", default=None, show_default=False,
              help="Exact output file path for the overlay image (e.g., output.png)")
@click.option("--output-base", "output_base", default=".", show_default=True,
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              help="Directory for auto-generated output file if --output is not set")
@click.option("--logging/--no-logging", default=True, show_default=True, help="Enable verbose logs")
def main(npy_a: str, npy_b: str, image_path: str, hf_model_name: str, metric: str,
         alpha: float, percentile: float, output_path: Optional[str], output_base: str,
         logging: bool) -> None:
    """Compare two patch-level embedding .npy files and overlay a red heatmap on the image.

    This version ALWAYS:
      - strips the first and last rows ([vision_start], [vision_end]) from the embeddings
      - applies a 2x2 merge (halves nx and ny from the processor grid)
      - colors by normalized diffs, labels top-K with RAW metric values
    """

    # Load arrays
    a = np.load(npy_a)
    b = np.load(npy_b)

    if a.shape != b.shape:
        raise click.ClickException(f".npy shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim == 1:
        raise click.ClickException("Expected patch-level embeddings (2D arrays), got pooled 1D vectors")
    if a.shape[0] < 3:
        raise click.ClickException("Not enough rows to strip [vision_start]/[vision_end].")

    # ALWAYS strip [vision_start] (row 0) and [vision_end] (row -1)
    a = a[1:-1, :]
    b = b[1:-1, :]
    if logging:
        click.echo(f"After stripping vision delimiters: npy shape {a.shape}")

    # Compute RAW differences AFTER stripping
    diffs_raw = _compute_diffs(a, b, metric.lower())  # (num_patches_eff,)

    # Infer raw grid from image
    nx, ny = _infer_grid_from_image(image_path, hf_model_name)

    # ALWAYS apply 2x2 merge (per-axis factor=2)
    if nx % 2 != 0 or ny % 2 != 0:
        raise click.ClickException(
            f"Cannot apply 2x2 merge to grid {nx}x{ny} (both dimensions must be even)."
        )
    eff_nx = nx // 2
    eff_ny = ny // 2
    expected = eff_nx * eff_ny

    # Check patch count after stripping matches merged grid
    if diffs_raw.shape[0] != expected:
        raise click.ClickException(
            f"Patch count mismatch. Heatmap patches={diffs_raw.shape[0]} but expected "
            f"{eff_nx}*{eff_ny}={expected} (raw grid {nx}*{ny} merged 2x2)."
        )

    # Percentile clip + min-max normalize for coloring ONLY
    if percentile is not None and 0 < percentile <= 100:
        p = float(np.percentile(diffs_raw, percentile))
        diffs_clipped = np.clip(diffs_raw, 0.0, p)
    else:
        diffs_clipped = diffs_raw

    dmin = float(diffs_clipped.min()) if diffs_clipped.size else 0.0
    dmax = float(diffs_clipped.max()) if diffs_clipped.size else 0.0
    if dmax > dmin:
        diffs_norm = (diffs_clipped - dmin) / (dmax - dmin + 1e-12)
    else:
        diffs_norm = np.zeros_like(diffs_clipped, dtype=np.float32)

    # Build overlay (colors from normalized, labels show RAW metric)
    with Image.open(image_path).convert("RGBA") as base_img:
        w, h = base_img.size
        overlay = _make_overlay(
            heatmap_values=diffs_norm,
            raw_values=diffs_raw,
            nx=eff_nx,
            ny=eff_ny,
            img_w=w,
            img_h=h,
            alpha=alpha,
            topk=10,  # adjust here if you want more/less labels (no CLI arg)
        )
        composed = Image.alpha_composite(base_img, overlay)

        # Decide output path
        if output_path is None or not str(output_path).strip():
            os.makedirs(output_base, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            rnd = uuid.uuid4().hex[:8]
            filename = f"{stem}-heatmap-{metric}-rawlabels-merged2x2-{ts}-{rnd}.png"
            final_path = os.path.join(output_base, filename)
        else:
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            final_path = output_path

        composed.save(final_path)

    click.echo(f"Saved heatmap overlay to {os.path.abspath(final_path)}")


if __name__ == "__main__":
    main()  # type: ignore
