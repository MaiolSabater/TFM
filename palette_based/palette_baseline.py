
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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    nnfm_loss_fn = NNFMLoss(device='cuda')
    if args.point_cloud:
        xyz, o, s = gaussians.load_ply(args.point_cloud, reset_basis_dim=args.reset_basis_dim)
        original_xyz, original_opacity, original_scale = torch.tensor(xyz).cuda(), torch.tensor(o).cuda(), torch.tensor(s).cuda()
        first_iter = 30_000

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


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

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print(vars(lp.extract(args)))
    # print("Optimizing " + args.model_path)

    # # Initialize system state (RNG)
    # safe_state(args.quiet)

    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # # All done
    # print("\nTraining complete.")
