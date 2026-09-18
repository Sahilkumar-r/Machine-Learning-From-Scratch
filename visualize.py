"""
visualize.py - Display before/after comparison of processed images.

Run this AFTER your image_processor.py has finished.
It reads from images/ (originals) and output/ (edge-detected results),
then builds a side-by-side grid and saves it to output/.

Usage:
    python visualize.py              # all discovered batch subdirectories
    python visualize.py batch_A      # one specific batch
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont


def discover_batches(images_root="images"):
    """Return sorted subdirectory names under images_root that contain PNG files."""
    if not os.path.isdir(images_root):
        return []
    batches = []
    for name in sorted(os.listdir(images_root)):
        path = os.path.join(images_root, name)
        if os.path.isdir(path) and any(
            f.lower().endswith(".png") for f in os.listdir(path)
        ):
            batches.append(name)
    return batches


def make_comparison_grid(batch_name, n_cols=5):
    """
    Build a grid showing original (top row) vs edge-detected (bottom row)
    for the first n_cols images of a batch. Saves the result as a PNG.
    """
    input_dir  = os.path.join("images",  batch_name)
    output_dir = os.path.join("output", batch_name)

    if not os.path.exists(input_dir):
        print(f"  [skip] Input folder not found: {input_dir}")
        return
    if not os.path.exists(output_dir):
        print(f"  [skip] Output folder not found: {output_dir} — run image_processor.py first")
        return

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.png'))[:n_cols]
    if not files:
        print(f"  [skip] No PNG files in {input_dir}")
        return

    sample = Image.open(os.path.join(input_dir, files[0]))
    W, H = sample.size

    PAD   = 8      # pixels between images
    LABEL = 22     # height of text label above each row
    COLS  = len(files)

    grid_w = PAD + COLS * (W + PAD)
    grid_h = PAD + 2 * (LABEL + H + PAD)
    grid   = Image.new('RGB', (grid_w, grid_h), color=(40, 40, 40))
    draw   = ImageDraw.Draw(grid)

    row_labels = ["Original", "Edge Detection"]

    for row, label in enumerate(row_labels):
        y_label = PAD + row * (LABEL + H + PAD)
        y_img   = y_label + LABEL
        draw.text((PAD, y_label), f"── {label} ({batch_name}) ──",
                  fill=(200, 200, 200))

        for col, filename in enumerate(files):
            x = PAD + col * (W + PAD)

            if row == 0:
                src_path = os.path.join(input_dir, filename)
                tile = Image.open(src_path).convert('RGB')
            else:
                edge_path = os.path.join(output_dir, filename)
                if not os.path.exists(edge_path):
                    tile = Image.new('RGB', (W, H), color=(80, 40, 40))
                    ImageDraw.Draw(tile).text((5, H // 2 - 8),
                                             "not found", fill=(255, 100, 100))
                else:
                    tile = Image.open(edge_path).convert('RGB')

            grid.paste(tile, (x, y_img))

    out_path = os.path.join("output", f"comparison_{batch_name}.png")
    grid.save(out_path)
    print(f"  Saved -> {out_path}")
    grid.show()


def main():
    batches = sys.argv[1:] if len(sys.argv) > 1 else discover_batches()
    if not batches:
        print("No batch subdirectories with PNG files found under images/")
        return
    print("Building visual comparisons...")
    for b in batches:
        print(f"\n  Batch: {b}")
        make_comparison_grid(b)
    print("\nDone.  Check output/ for comparison_*.png files.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
