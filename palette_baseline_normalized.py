
import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, StyleOptimizationParams
from utils.nnfm_loss import NNFMLoss, match_colors_for_image_set, color_histgram_match
from utils.image_utils import load_and_preprocess_style_image
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from compute_color_palette import extract_palette
from itertools import product
from skimage.color import rgb2lab
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

SH_C0 = 0.28209479177387814

def set_geometry_grad(gaussian_model,freeze):
    if freeze:
        gaussian_model._xyz.requires_grad = False
        gaussian_model._scaling.requires_grad = False
        gaussian_model._rotation.requires_grad = False
        gaussian_model._opacity.requires_grad = False
    else:
        gaussian_model._xyz.requires_grad = True
        gaussian_model._scaling.requires_grad = True
        gaussian_model._rotation.requires_grad = True
        gaussian_model._opacity.requires_grad = True

def to_lab(rgb):
    return rgb2lab(rgb.reshape(1,1,3)).reshape(3)
    

def decompose(gaussian_colors, palette):
    colors = np.array(gaussian_colors)
    basis  = np.array(palette)

    labs_basis  = np.array([to_lab(c) for c in basis])
    labs_colors = np.array([to_lab(c) for c in colors])
    dists = np.linalg.norm(labs_colors[:, None, :] - labs_basis[None, :, :], axis=2)
    assigned = np.argmin(dists, axis=1)

    b_j = basis[assigned]
    cos_theta = np.sum(colors * b_j, axis=1) / (
        np.linalg.norm(colors, axis=1) * np.linalg.norm(b_j, axis=1) + 1e-8
    )
    d_mag = np.linalg.norm(colors, axis=1)
    b_j_mag = np.linalg.norm(b_j, axis=1)
    d_normalized = d_mag / (b_j_mag + 1e-8)  # relative magnitude

    return assigned, cos_theta, d_normalized


def recolor(assigned, cos_theta, d_normalized, new_palette):
    new_basis = np.array(new_palette)
    b_j_prime = new_basis[assigned]  # (N, 3) — NOT normalized

    # d_normalized already relative, b_j_prime carries its own magnitude
    new_colors = cos_theta[:, None] * d_normalized[:, None] * b_j_prime

    return np.clip(new_colors, 0, 1)
def make_palette_strip(palette, width, height=40):
    """Return a (height, width, 3) uint8 array of palette swatches."""
    k = len(palette)
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    swatch_w = width // k
    for i, color in enumerate(palette):
        x0 = i * swatch_w
        x1 = x0 + swatch_w if i < k - 1 else width
        strip[:, x0:x1] = (np.clip(color, 0, 1) * 255).astype(np.uint8)
    return strip


def make_report(orig_np, recolored_np, source_palette, target_palette):
    """
    Returns a single (H*2 + strip_h, W*2, 3) uint8 image:
      top row:    original | recolored
      bottom row: source palette | target palette
    """
    H, W = orig_np.shape[:2]
    src_strip = make_palette_strip(source_palette, W)
    tgt_strip = make_palette_strip(target_palette, W)
    top = np.concatenate([orig_np, recolored_np], axis=1)
    bottom = np.concatenate([src_strip, tgt_strip], axis=1)
    return np.concatenate([top, bottom], axis=0)


def training(dataset, opt, pipe, checkpoint, source_palette, target_palette, point_cloud=None, reset_basis_dim=1, save_report=False):

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    if point_cloud:
        gaussians.load_ply(point_cloud, reset_basis_dim=reset_basis_dim)

    # Closed-form palette recoloring
    S0 = gaussians.get_base_color  # (N, 3) RGB in [0, 1]
    assigned, cos_theta, d_mag = decompose(S0.detach().cpu().numpy(), source_palette)
    print(f"Assigned clusters: {assigned}")
    new_S0 = recolor(assigned, cos_theta, d_mag, target_palette)

    new_dc = (torch.tensor(new_S0, dtype=torch.float32, device="cuda") - 0.5) / SH_C0
    gaussians._features_dc.data[:, 0, :] = new_dc

    # Save recolored Gaussians
    scene.save(0)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    out_dir = os.path.join(dataset.model_path, "recolored_sm")
    os.makedirs(out_dir, exist_ok=True)

    report_dir = os.path.join(dataset.model_path, "reports")
    if save_report:
        os.makedirs(report_dir, exist_ok=True)

    with torch.no_grad():
        for view in tqdm(scene.getTrainCameras(), desc="Rendering recolored views"):
            render_pkg = render(view, gaussians, pipe, background)
            image = render_pkg["render"]  # (3, H, W)
            img_np = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_dir, f"{view.image_name}.png"), img_np)

            if save_report:
                orig_np = (view.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                report = make_report(orig_np, img_np, source_palette, target_palette)
                imageio.imwrite(os.path.join(report_dir, f"{view.image_name}.png"), report)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = StyleOptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # style params
    parser.add_argument("--point_cloud", type=str, help='trained real 3DGS ply', default = None)
    parser.add_argument("--style", type=str, help="path to style image")
    parser.add_argument("--second_style", type=str, default="", help="path to second style image")
    parser.add_argument("--style_weight", type=float, default=5, help="style loss weight")
    parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
    parser.add_argument("--img_tv_weight", type=float, default=1, help="image tv loss weight")
    parser.add_argument(
        "--vgg_block",
        type=list,
        default=[2,3],
        help="vgg block for nnfm extracting feature maps",
    )
    parser.add_argument(
        "--reset_basis_dim",
        type=int,
        default=1,
        help="whether to reset the number of spherical harmonics basis to this specified number",
    )
    parser.add_argument("--preserve_color", action="store_true", default=False)
    parser.add_argument("--scale_level", type=int, default=None, choices=[0,1,2], help='the scale of style pattern, can be [0,1,2]')
    parser.add_argument("--mask_dir", default=None, type=str, help="The directory of multiview masks")
    parser.add_argument("--save_report", action="store_true", default=False, help="Save per-view comparison reports")

    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    images_path = os.path.join(dataset.source_path, dataset.images)

    print(args.style)
    source_palette, _, _ = extract_palette(image_path=images_path, n_imgs=5)
    target_palette, _, _ = extract_palette(image_path=args.style)

    print("Recoloring " + dataset.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        dataset, op.extract(args), pp.extract(args),
        args.start_checkpoint, source_palette, target_palette,
        point_cloud=args.point_cloud, reset_basis_dim=args.reset_basis_dim,
        save_report=args.save_report,
    )
