"""
example_usage.py - Basic example of using image_filters.py

Shows how to call each function. Run this before starting the project
to make sure everything works on your machine.
"""

import os
from image_filters import (
    load_image,
    to_grayscale,
    apply_blur,
    detect_edges,
    compute_stats,
    save_image,
)

# ── Load one image ────────────────────────────────────────────────────────────
img = load_image("images/batch_A/img_001.png")
print(f"Loaded image: size={img.size}, mode={img.mode}")

# ── Convert to grayscale ──────────────────────────────────────────────────────
gray = to_grayscale(img)
print(f"Grayscale mode: {gray.mode}")  # 'L'

# ── Apply blur ────────────────────────────────────────────────────────────────
blurred = apply_blur(gray, radius=2)
print("Applied Gaussian blur (radius=2)")

# ── Detect edges ──────────────────────────────────────────────────────────────
edges = detect_edges(img)
print(f"Edge detection done: mode={edges.mode}")  # 'L'

# ── Compute statistics ────────────────────────────────────────────────────────
stats = compute_stats(img)
print("\nImage statistics:")
print(f"  mean_brightness : {stats['mean_brightness']:.1f}   (0=black, 255=white)")
print(f"  std_brightness  : {stats['std_brightness']:.1f}   (higher = more contrast)")
print(f"  edge_density    : {stats['edge_density']:.4f}  "
      f"({stats['edge_density'] * 100:.1f}% of pixels are edges)")

# ── Save output ───────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
save_image(edges, "output/example_edges.png")
print("\nSaved edge-detected image -> output/example_edges.png")


# ── Display an image on screen ──────────────────────────────────────────────────
# Want to actually SEE the result? Any PIL image has a .show() method that
# opens it in your computer's image viewer:
#
#     img.show()      # the original image
#     edges.show()    # the edge-detected result
#
# (We leave these commented out so the script doesn't pop up windows every run.)
#
# There is also a small helper, visualize.py, that puts the original and the
# edge image side by side so the change is easy to see. Run it after your
# image_processor.py has produced results:
#
#     python visualize.py            # all batches
#     python visualize.py batch_A    # one batch

print("\nTip: call img.show() or edges.show() to view an image,")
print("     or run visualize.py to see a before/after comparison.")
