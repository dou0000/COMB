"""
COMB whole-slide-image inference -- general purpose.

Runs seamless WSI virtual staining with the Consistency Memory Bank (COMB).
Modality-agnostic: any number of spectral input channels (auto-detected from
the data or set with --num_channels). Pass --variant to toggle COMB components
for ablation studies.

Quick usage
-----------
    # Full COMB (paper default)
    python WSI_inference.py \\
        --ckpt_path checkpoint_net_G.pth \\
        --input my_roi.npy \\
        --output out.png

    # Directory of ROIs
    python WSI_inference.py \\
        --ckpt_path checkpoint_net_G.pth \\
        --input ./rois/ \\
        --output ./out/

    # Ablation: turn off the neighbor-aware CBAM only
    python WSI_inference.py ... --variant no_cbam

    # Paper-faithful CBAM mixing (alpha=0.3, exclude center)
    python WSI_inference.py ... --alpha 0.3 --exclude_center

CLI variants
------------
    ours      use_halo_banks=True,  use_cbam_bank=True   (full COMB)
    no_cbam   use_halo_banks=True,  use_cbam_bank=False
    no_halo   use_halo_banks=False, use_cbam_bank=True
    baseline  use_halo_banks=False, use_cbam_bank=False  (plain model)
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.networks.comb_inference import (
    AttUnetGenerator,
    init_prefetch_margin,
    assert_no_raw_padding,
    MarginBankConv,
    AnchorBankChannelAttention,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


VARIANT_FLAGS = {
    "baseline": (False, False),
    "no_cbam":  (True,  False),
    "no_halo":  (False, True),
    "ours":     (True,  True),
}


# ============================================================================
# Generic dataset: each file is one ROI of any number of spectral channels.
# Supports .npy and .mat. Auto-detects (C, H, W) vs (H, W, C).
# ============================================================================

class WSIInputDataset(Dataset):
    def __init__(self, paths, num_channels=None, clip_max=1.0):
        self.paths = [Path(p) for p in paths]
        self.num_channels = num_channels
        self.clip_max = float(clip_max)
        if not self.paths:
            raise RuntimeError("No input files provided")

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        suffix = path.suffix.lower()
        if suffix == ".npy":
            img = np.load(path).astype(np.float32)
        elif suffix == ".mat":
            try:
                import scipy.io
                mat = scipy.io.loadmat(path)
                keys = [k for k in mat.keys() if not k.startswith("__")]
                img = np.asarray(mat[keys[0]]).astype(np.float32)
            except (NotImplementedError, ValueError):
                import h5py
                with h5py.File(path, "r") as f:
                    keys = list(f.keys())
                    img = np.asarray(f[keys[0]][:]).astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported file format: {suffix}")
        img = np.nan_to_num(img, nan=0.0)
        return img

    def __getitem__(self, index):
        path = self.paths[index]
        img = self._load(path)

        # normalize shape -> (C, H, W)
        if img.ndim == 3 and img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
            img = np.transpose(img, (2, 0, 1))     # (H, W, C) -> (C, H, W)

        # match requested channel count
        if self.num_channels is not None:
            C = int(self.num_channels)
            if img.shape[0] > C:
                img = img[:C]
            elif img.shape[0] < C:
                pad = np.zeros((C - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
                img = np.concatenate((img, pad), axis=0)

        img = np.clip(img, 0, self.clip_max) / max(self.clip_max, 1e-9)
        return {"x": img, "name": path.stem, "channels": int(img.shape[0])}


# ============================================================================
# Helpers
# ============================================================================

def override_cbam_bank_params(net, alpha=None, include_center=None):
    """Patch every AnchorBankChannelAttention's hyperparameters after construction."""
    n = 0
    for m in net.modules():
        if isinstance(m, AnchorBankChannelAttention):
            if alpha is not None:
                m.alpha = float(alpha)
            if include_center is not None:
                m.include_center = bool(include_center)
            n += 1
    return n


def set_conveyor_belt(model, radius):
    """Set the sliding-window eviction width on every banked layer."""
    window_size = int(radius + 2)
    for m in model.modules():
        if isinstance(m, (MarginBankConv, AnchorBankChannelAttention)):
            m.keep_rows_up = window_size
            m.keep_rows_down = window_size


def make_row_col_positions(r, c, tile, stride):
    rs = []
    s = 0
    while True:
        e = s + tile
        if e > r:
            e = r
            s = max(e - stride, 0)
        rs.append((s, e))
        if e >= r:
            break
        s += stride
    cs = []
    s = 0
    while True:
        e = s + tile
        if e > c:
            e = c
            s = max(e - stride, 0)
        cs.append((s, e))
        if e >= c:
            break
        s += stride
    return rs, cs


def build_generator(opt, use_halo, use_cbam, norm_layer, num_channels, device):
    net = AttUnetGenerator(
        opt=None,
        norm_layer=norm_layer,
        fpn_feature="decoder",
        num_channels=num_channels,
        use_halo_banks=use_halo,
        use_cbam_bank=use_cbam,
        bank_device=opt.bank_device,
        cbam_radius=opt.cbam_radius,
    )
    if use_cbam:
        n = override_cbam_bank_params(
            net, alpha=opt.alpha, include_center=opt.include_center,
        )
        print(f"[cbam-bank] overrode {n} module(s) with alpha={opt.alpha}, "
              f"include_center={opt.include_center}")
    ckpt = torch.load(opt.ckpt_path, map_location="cpu")
    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"[ckpt] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[ckpt] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"[ckpt] loaded {opt.ckpt_path}")

    net.encoder.eval()
    net.to(device).eval()
    net.to(memory_format=torch.channels_last)
    net.half()
    return net


# ============================================================================
# Main inference loop
# ============================================================================

def run_inference(opt, dataset, device, netG, anchor_enabled):
    MACRO_TILE = opt.tile
    OVERLAP = opt.overlap
    CROP = OVERLAP // 2
    BIG_STRIDE = MACRO_TILE - OVERLAP
    scaling_factor = float(opt.scaling_factor)
    R = opt.cbam_radius

    out_dir = Path(opt.output)
    if len(dataset) > 1 or out_dir.suffix not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        out_dir.mkdir(parents=True, exist_ok=True)
        per_file_out = lambda name: out_dir / f"{name}_full_res.png"
    else:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        per_file_out = lambda name: out_dir

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=opt.n_cpu,
                            pin_memory=(device.type == "cuda"))

    with torch.inference_mode():
        for batch in dataloader:
            x = batch["x"].float()
            name = batch["name"][0]
            t0 = time.time()

            b, bands, r_raw, c_raw = x.shape
            r = int(r_raw / scaling_factor + 1)
            c = int(c_raw / scaling_factor + 1)
            rn = int(np.ceil(r / MACRO_TILE))
            cn = int(np.ceil(c / MACRO_TILE))
            fhr = np.zeros((b, 3, rn * MACRO_TILE + 1, cn * MACRO_TILE + 1), dtype=np.float32)

            row_positions, col_positions = make_row_col_positions(r, c, MACRO_TILE, BIG_STRIDE)
            Y, X = len(row_positions), len(col_positions)
            all_patches = [(ii, jj) for jj in range(X) for ii in range(Y)]
            total_patches = len(all_patches)
            PIPELINE_LAG = min(Y * R + R, total_patches)

            if anchor_enabled:
                set_conveyor_belt(netG, radius=R)
                init_prefetch_margin(netG, y_anchor_num=Y, x_anchor_num=X)

            print(f"\n=== {name}  ({opt.variant}) ===")
            print(f"  input: {tuple(x.shape)}, output canvas: {r} x {c}")
            print(f"  grid {Y}x{X} | R={R} | lag {PIPELINE_LAG}")

            def build_tile(ii, jj, kk):
                rstart, rend = row_positions[ii]
                cstart, cend = col_positions[jj]
                rs = int(rstart * scaling_factor); re_ = int(rend * scaling_factor)
                cs = int(cstart * scaling_factor); ce = int(cend * scaling_factor)
                xx = x[kk:kk+1, :, rs:re_, cs:ce]
                pad_r = max(int(MACRO_TILE * scaling_factor) - xx.shape[2], 0)
                pad_c = max(int(MACRO_TILE * scaling_factor) - xx.shape[3], 0)
                if pad_r > 0 or pad_c > 0:
                    xx = F.pad(xx, (0, pad_c, 0, pad_r), mode="constant", value=0)
                xx = xx.to(device=device, dtype=torch.float16, non_blocking=True)
                xx = F.interpolate(xx, size=(MACRO_TILE, MACRO_TILE), mode="bicubic", align_corners=False)
                tile = torch.zeros(1, bands, MACRO_TILE, MACRO_TILE, device=device, dtype=torch.float16)
                tile[0, :bands] = xx[0]
                return tile, (rstart, rend, cstart, cend)

            def write_back(out, kk, roi):
                rstart, rend, cstart, cend = roi
                a = out.detach().float().cpu().numpy() * 0.5 + 0.5
                if CROP > 0:
                    a = a[..., CROP:-CROP, CROP:-CROP]
                core_h = rend - rstart - 2 * CROP
                core_w = cend - cstart - 2 * CROP
                a = a[..., :core_h, :core_w]
                fhr[kk, :, rstart + CROP: rstart + CROP + core_h,
                            cstart + CROP: cstart + CROP + core_w] = a

            netG.eval()
            for kk in range(b):
                if anchor_enabled:
                    init_prefetch_margin(netG, y_anchor_num=Y, x_anchor_num=X)
                    print(f"  [Warmup] {PIPELINE_LAG} tiles...")
                    for k in range(PIPELINE_LAG):
                        tile_gpu, _ = build_tile(*all_patches[k], kk)
                        py, px = all_patches[k]
                        _ = netG.forward_prefetch(
                            tile_gpu, y_anchor=[py], x_anchor=[px],
                            pre_y_anchor=None, pre_x_anchor=None, overlap_px=OVERLAP,
                        )

                    print(f"  [Main] pipeline...")
                    for k in range(total_patches):
                        ay, ax = all_patches[k]
                        anchor, anchor_roi = build_tile(ay, ax, kk)
                        pfi = k + PIPELINE_LAG
                        if pfi < total_patches:
                            py, px = all_patches[pfi]
                            prefetch, _ = build_tile(py, px, kk)
                            inp = torch.cat((anchor, prefetch), dim=0)
                            out_batch = netG.forward_prefetch(
                                inp, y_anchor=[ay], x_anchor=[ax],
                                pre_y_anchor=[py], pre_x_anchor=[px], overlap_px=OVERLAP,
                            )
                            out = out_batch[0]
                        else:
                            out = netG.forward_prefetch(
                                anchor, y_anchor=[ay], x_anchor=[ax],
                                pre_y_anchor=None, pre_x_anchor=None, overlap_px=OVERLAP,
                            )
                            if out.dim() == 4:
                                out = out[0]
                        write_back(out, kk, anchor_roi)
                        if k % 100 == 0:
                            print(f"    tile {k}/{total_patches}")
                else:
                    print(f"  [Baseline] plain per-tile forward...")
                    for k in range(total_patches):
                        ay, ax = all_patches[k]
                        tile, roi = build_tile(ay, ax, kk)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = netG.forward(tile)
                        if out.dim() == 4:
                            out = out[0]
                        write_back(out, kk, roi)

            img = np.uint8(np.transpose(fhr, (0, 2, 3, 1)) * 255.0)[0, :r, :c, :]
            out_path = per_file_out(name)
            Image.fromarray(img).save(out_path)
            print(f"[Saved] {out_path}  (took {time.time()-t0:.2f}s)")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="COMB seamless WSI inference.")
    p.add_argument("--ckpt_path", required=True, help="Trained generator checkpoint (.pth)")
    p.add_argument("--input",     required=True, help="Single .npy/.mat file OR a directory of them")
    p.add_argument("--output",    required=True, help="Output PNG path (single file) OR directory")

    p.add_argument("--variant", default="ours", choices=list(VARIANT_FLAGS.keys()),
                   help="ours (full COMB), no_cbam, no_halo, baseline")
    p.add_argument("--num_channels", type=int, default=None,
                   help="Number of input spectral channels (auto-detect if omitted)")
    p.add_argument("--clip_max", type=float, default=1.0,
                   help="Input normalization clip max (default 1.0; e.g. 3.5 for SRS, 6.0 for IR)")
    p.add_argument("--scaling_factor", type=float, default=1.0,
                   help="Input/output resolution ratio (1.0 if same; 0.25 if input is 4x lower-res)")
    p.add_argument("--tile",          type=int, default=512, help="Tile size at output resolution")
    p.add_argument("--overlap",       type=int, default=16,  help="Tile overlap in pixels")
    p.add_argument("--cbam_radius",   type=int, default=2,   help="Neighbor radius R (paper R=2)")
    p.add_argument("--alpha",         type=float, default=0.3,
                   help="CBAM neighbor mixing weight: hat_g = (1-alpha)*g_local + alpha*g_neighbor. "
                        "Paper default (Eq. 2): 0.3.")
    p.add_argument("--include_center", action="store_true",
                   help="Include the center anchor in the neighbor average. Paper Eq 2 excludes the "
                        "center; default behaviour is paper-faithful (excluded). Pass this flag for "
                        "the code-default behaviour.")
    p.add_argument("--norm_layer",     default="batch", choices=["batch", "instance"])
    p.add_argument("--bank_device",    default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--n_cpu",          type=int, default=4)
    opt = p.parse_args()

    use_halo, use_cbam = VARIANT_FLAGS[opt.variant]
    anchor_enabled = use_halo or use_cbam

    norm_layer = nn.BatchNorm2d if opt.norm_layer == "batch" else nn.InstanceNorm2d
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # gather input file paths
    in_path = Path(opt.input)
    if in_path.is_dir():
        paths = sorted(list(in_path.glob("*.npy")) + list(in_path.glob("*.mat")))
        if not paths:
            raise RuntimeError(f"No .npy or .mat files in {in_path}")
    else:
        paths = [in_path]

    # auto-detect channels if not specified
    num_channels = opt.num_channels
    if num_channels is None:
        peek = WSIInputDataset([paths[0]], num_channels=None, clip_max=opt.clip_max)[0]
        num_channels = peek["channels"]
        print(f"[Auto-detect] num_channels = {num_channels}")

    dataset = WSIInputDataset(paths, num_channels=num_channels, clip_max=opt.clip_max)

    netG = build_generator(opt, use_halo, use_cbam, norm_layer, num_channels, device)
    if use_halo:
        assert_no_raw_padding(netG)

    print(f"[Variant] {opt.variant}  use_halo={use_halo}  use_cbam={use_cbam}")
    print(f"[Data]    {len(dataset)} ROI(s) at {num_channels} channels")
    run_inference(opt, dataset, device, netG, anchor_enabled)


if __name__ == "__main__":
    main()
