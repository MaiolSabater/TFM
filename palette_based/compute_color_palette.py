"""
Palette extraction from one or more images of the same scene.
- Builds a weighted color histogram in Lab space (16x16x16 bins)
- Runs k-means for k=2..10, computes inertia
- Finds optimal k via elbow method (max perpendicular distance)
- Extracts final palette, plots elbow curve and swatches
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.color import rgb2lab, lab2rgb
import argparse
import sys


# ── Histogram ────────────────────────────────────────────────────────────────

def build_histogram(pixels_rgb: np.ndarray, bins: int = 16):
    """
    Bin pixels into a bins^3 RGB histogram, compute per-bin mean color in Lab.
    Returns list of dicts: {rgb, lab, weight}
    pixels_rgb: (N, 3) float32 in [0, 1]
    """
    indices = np.floor(pixels_rgb * bins).clip(0, bins - 1).astype(np.int32)
    keys = indices[:, 0] * bins * bins + indices[:, 1] * bins + indices[:, 2]

    hist = {}
    for i, k in enumerate(keys):
        if k not in hist:
            hist[k] = {"sum": np.zeros(3), "count": 0}
        hist[k]["sum"] += pixels_rgb[i]
        hist[k]["count"] += 1

    result = []
    for entry in hist.values():
        rgb = entry["sum"] / entry["count"]
        lab = rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)
        result.append({"rgb": rgb, "lab": lab, "weight": entry["count"]})
    return result


# ── Deterministic k-means init (paper method) ────────────────────────────────

def init_means(bins, k, sigma_a=80.0):
    """
    Deterministic init: pick the heaviest bin, then attenuate nearby weights
    and pick the next heaviest. Run for k+1 steps, discard last (black anchor).
    """
    weights = np.array([b["weight"] for b in bins], dtype=float)
    labs = np.array([b["lab"] for b in bins])
    chosen = []

    for _ in range(k + 1):
        idx = int(np.argmax(weights))
        chosen.append(idx)
        dists = np.linalg.norm(labs - labs[idx], axis=1)
        weights *= 1.0 - np.exp(-(dists ** 2) / (sigma_a ** 2))

    # discard last (black anchor), return lab coords of chosen
    return labs[chosen[:k]].copy()


# ── Weighted k-means in Lab ───────────────────────────────────────────────────

def kmeans_lab(bins, k, max_iter=100):
    """
    Weighted k-means on bin Lab colors.
    Returns: palette_rgb (k, 3), inertia (float)
    """
    labs = np.array([b["lab"] for b in bins])       # (M, 3)
    weights = np.array([b["weight"] for b in bins])  # (M,)
    means = init_means(bins, k)                       # (k, 3)

    assignments = np.zeros(len(bins), dtype=int)

    for _ in range(max_iter):
        # assign
        dists = np.linalg.norm(labs[:, None, :] - means[None, :, :], axis=2)  # (M, k)
        new_assignments = np.argmin(dists, axis=1)

        if np.all(new_assignments == assignments):
            break
        assignments = new_assignments

        # update means
        for j in range(k):
            mask = assignments == j
            if mask.any():
                w = weights[mask]
                means[j] = (labs[mask] * w[:, None]).sum(axis=0) / w.sum()

    # inertia: weighted sum of squared distances to assigned mean
    assigned_means = means[assignments]
    sq_dists = np.sum((labs - assigned_means) ** 2, axis=1)
    inertia = float((weights * sq_dists).sum())

    # convert means back to RGB, clamp to [0, 1]
    palette_rgb = []
    for m in means:
        rgb = lab2rgb(m.reshape(1, 1, 3)).reshape(3)
        palette_rgb.append(np.clip(rgb, 0, 1))

    return np.array(palette_rgb), inertia


# ── Elbow method ─────────────────────────────────────────────────────────────

def find_elbow(inertias):
    """
    Max perpendicular distance from the line connecting first and last point.
    Returns the best k (1-indexed into the inertias list, starting at k=2).
    """
    n = len(inertias)
    if n < 3:
        return n  # can't find elbow with fewer than 3 points

    p1 = np.array([0.0, inertias[0]])
    p2 = np.array([float(n - 1), inertias[-1]])
    line = p2 - p1
    line_len = np.linalg.norm(line)

    max_dist, elbow_idx = -1.0, 0
    for i in range(1, n - 1):
        p = np.array([float(i), inertias[i]])
        dist = abs(np.cross(line, p1 - p)) / line_len
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    return elbow_idx + 2  # +2 because k starts at 2


# ── Image loading ─────────────────────────────────────────────────────────────

def load_pixels(paths, max_samples_per_image=50_000):
    """
    Load and sample pixels from one or more images.
    Returns (N, 3) float32 array in [0, 1].
    """
    all_pixels = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        flat = arr.reshape(-1, 3)
        if len(flat) > max_samples_per_image:
            idx = np.random.choice(len(flat), max_samples_per_image, replace=False)
            flat = flat[idx]
        all_pixels.append(flat)
        print(f"  Loaded {p.name}: {img.size[0]}×{img.size[1]}, sampled {len(flat)} px")
    return np.concatenate(all_pixels, axis=0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(inertias, palette_rgb, chosen_k, save_path=None):
    k_values = list(range(2, 2 + len(inertias)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Palette extraction", fontsize=13, fontweight="bold")

    # Elbow curve
    ax = axes[0]
    ax.plot(k_values, inertias, marker="o", color="#378ADD", linewidth=1.5, markersize=5)
    ax.scatter([chosen_k], [inertias[chosen_k - 2]], color="#D85A30", s=80, zorder=5,
               label=f"Elbow k={chosen_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia (Lab space)")
    ax.set_title("Elbow method")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Palette swatches
    ax = axes[1]
    ax.set_xlim(0, chosen_k)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Palette (k={chosen_k})")

    swatch_w = 1.0
    for i, color in enumerate(palette_rgb):
        rect = patches.FancyBboxPatch(
            (i * swatch_w + 0.05, 0.15), swatch_w - 0.1, 0.7,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=1.5
        )
        ax.add_patch(rect)
        hex_color = "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in color])
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "white" if brightness < 0.5 else "black"
        ax.text(i + 0.5, 0.08, hex_color, ha="center", va="center",
                fontsize=8, color="gray", fontfamily="monospace")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_palette(image_paths, k_override=None, bins=16, save_plot=None):
    """
    Full pipeline: load → histogram → elbow → palette.

    Args:
        image_paths:  list of Path or str
        k_override:   if set, skip elbow and use this k
        bins:         histogram bins per channel (default 16)
        save_plot:    path to save the result plot (None = show interactively)

    Returns:
        palette_rgb:  (k, 3) numpy array, colors in [0, 1]
        chosen_k:     int
        inertias:     list of inertias for k=2..10
    """
    paths = [Path(p) for p in image_paths]
    print(f"\nLoading {len(paths)} image(s)...")
    pixels = load_pixels(paths)
    print(f"Total pixels sampled: {len(pixels)}")

    print("Building histogram...")
    hist_bins = build_histogram(pixels, bins=bins)
    print(f"Histogram bins: {len(hist_bins)}")

    print("Running k-means for k=2..10...")
    inertias = []
    for k in range(2, 11):
        _, inertia = kmeans_lab(hist_bins, k)
        inertias.append(inertia)
        print(f"  k={k:2d}  inertia={inertia:.1f}")

    if k_override is not None:
        chosen_k = k_override
        print(f"\nUsing k={chosen_k} (override)")
    else:
        chosen_k = find_elbow(inertias)
        print(f"\nElbow at k={chosen_k}")

    print(f"Extracting final palette with k={chosen_k}...")
    palette_rgb, _ = kmeans_lab(hist_bins, chosen_k)

    print("\nPalette colors (RGB):")
    for i, c in enumerate(palette_rgb):
        hex_c = "#{:02x}{:02x}{:02x}".format(*[int(v * 255) for v in c])
        print(f"  b{i+1}: {hex_c}  rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})")

    plot_results(inertias, palette_rgb, chosen_k, save_path=save_plot)

    return palette_rgb, chosen_k, inertias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a color palette from one or more images.")
    parser.add_argument("images", nargs="+", help="Input image file(s)")
    parser.add_argument("--k", type=int, default=None, help="Override k (default: auto via elbow)")
    parser.add_argument("--bins", type=int, default=16, help="Histogram bins per channel (default: 16)")
    parser.add_argument("--save", type=str, default=None, help="Save plot to this path instead of displaying")
    args = parser.parse_args()

    palette, k, inertias = extract_palette(
        args.images,
        k_override=args.k,
        bins=args.bins,
        save_plot=args.save
    )