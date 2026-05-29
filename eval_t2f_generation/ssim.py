"""
Evaluate TaDiff FLAIR generation SSIM on MU-Glioma-Post test patients.

Protocol requested by user:
- Patients: use testing_patient_ids from train_mu_glioma_post_config.py
- Data source: mu_glioma_post_output/*.npy
- Target session: last session of each patient
- Slices: middle three slices [D//2 - 1, D//2, D//2 + 1]
- Run model 3 times per patient (one per slice), total 20*3 = 60 runs
- Keep only FLAIR modality from prediction and GT
- Stack 3 FLAIR slices and compute SSIM
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.train_mu_glioma_post_config import TrainMuGliomaPostConfig
from src.net.diffusion import GaussianDiffusion
from src.tadiff_model import Tadiff_model


def _center_crop_or_pad_2d(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    h, w = arr.shape[-2], arr.shape[-1]

    if h > out_h:
        top = (h - out_h) // 2
        arr = arr[..., top : top + out_h, :]
    if w > out_w:
        left = (w - out_w) // 2
        arr = arr[..., :, left : left + out_w]

    h, w = arr.shape[-2], arr.shape[-1]
    pad_h = max(out_h - h, 0)
    pad_w = max(out_w - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_spec = [(0, 0)] * arr.ndim
        pad_spec[-2] = (pad_top, pad_bottom)
        pad_spec[-1] = (pad_left, pad_right)
        arr = np.pad(arr, pad_spec, mode="constant")
    return arr


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter_nd(input: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    assert all(ws == 1 for ws in win.shape[1:-1]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)
    c = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=c)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} "
                f"and win size: {win.shape[-1]}",
            )
    return out


def _ssim_tadiff(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float,
    win: torch.Tensor,
    size_average: bool = True,
    k: tuple[float, float] = (0.01, 0.03),
) -> torch.Tensor:
    k1, k2 = k
    compensation = 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    win = win.to(device=x.device, dtype=x.dtype)
    mu1 = _gaussian_filter_nd(x, win)
    mu2 = _gaussian_filter_nd(y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * (_gaussian_filter_nd(x * x, win) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter_nd(y * y, win) - mu2_sq)
    sigma12 = compensation * (_gaussian_filter_nd(x * y, win) - mu1_mu2)
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    if size_average:
        return ssim_per_channel.mean()
    return ssim_per_channel.mean(1)


def tadiff_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: torch.Tensor | None = None,
    k: tuple[float, float] = (0.01, 0.03),
) -> torch.Tensor:
    if not x.shape == y.shape:
        raise ValueError(f"Input images should have the same dimensions, got {x.shape} and {y.shape}.")
    for d in range(len(x.shape) - 1, 1, -1):
        x = x.squeeze(dim=d).float()
        y = y.squeeze(dim=d).float()
    if len(x.shape) != 4:
        raise ValueError(f"Input images should be 4-d, but got {x.shape}")
    if not x.type() == y.type():
        raise ValueError(f"Input images should have the same dtype, but got {x.type()} and {y.type()}.")
    if win is not None:
        win_size = win.shape[-1]
    if win_size % 2 != 1:
        raise ValueError("Window size should be odd.")
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))
    return _ssim_tadiff(x, y, data_range=data_range, win=win, size_average=True, k=k)


def _independent_minmax_to_01(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Independent min-max normalization to [0,1] for pred and GT separately.

    This handles differing value ranges (e.g., pred often near [0,1], GT not).
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    a_min, a_max = float(np.min(a)), float(np.max(a))
    b_min, b_max = float(np.min(b)), float(np.max(b))

    if a_max - a_min > eps:
        a_01 = (a - a_min) / (a_max - a_min)
    else:
        a_01 = np.zeros_like(a, dtype=np.float32)

    if b_max - b_min > eps:
        b_01 = (b - b_min) / (b_max - b_min)
    else:
        b_01 = np.zeros_like(b, dtype=np.float32)

    return a_01, b_01


def _ssim_tadiff_from_numpy_chw(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
    """
    3-channel SSIM in one call: inputs are [3,H,W] in [0,1].
    """
    x = torch.from_numpy(pred_01).unsqueeze(0).float()
    y = torch.from_numpy(gt_01).unsqueeze(0).float()
    return float(tadiff_ssim(x, y, data_range=1.0, win_size=11, win_sigma=1.5).item())


def _load_eval_cfg_from_checkpoint(ckpt_path: Path) -> TrainMuGliomaPostConfig:
    """
    Combine local eval defaults with model architecture from checkpoint.

    This avoids shape mismatches when the checkpoint was trained with
    different model width/depth settings than current local defaults.
    """
    cfg = TrainMuGliomaPostConfig()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    ck_cfg = hp.get("config", None)
    if ck_cfg is None:
        return cfg

    ck_dict = dict(ck_cfg) if hasattr(ck_cfg, "items") else {}
    if not ck_dict:
        return cfg

    merge_keys = [
        "image_size",
        "in_channels",
        "out_channels",
        "model_channels",
        "num_res_blocks",
        "channel_mult",
        "attention_resolutions",
        "num_heads",
        "num_classes",
        "max_T",
        "ddpm_schedule",
        "aux_loss_w",
        "opt",
        "lr",
        "weight_decay",
        "warmup_steps",
        "max_steps",
        "max_epochs",
        "batch_size",
        "num_workers",
        "grad_clip",
        "accumulate_grad_batches",
        "precision",
    ]
    for k in merge_keys:
        if k in ck_dict:
            setattr(cfg, k, ck_dict[k])
    return cfg


def _save_patient_visualization(
    patient_id: str,
    slices: Sequence[int],
    pred_stack: Sequence[np.ndarray],
    gt_stack: Sequence[np.ndarray],
    output_dir: Path,
) -> None:
    """
    Save one PNG containing GT/Pred comparison for the 3 evaluated slices.
    Layout: 2 rows x 3 columns (top: GT, bottom: Pred).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for col, (z, pred, gt) in enumerate(zip(slices, pred_stack, gt_stack)):
        axes[0, col].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"GT FLAIR z={z}")
        axes[0, col].axis("off")

        axes[1, col].imshow(pred, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"Pred FLAIR z={z}")
        axes[1, col].axis("off")

    fig.suptitle(f"{patient_id} | 3-slice FLAIR comparison", fontsize=12)
    save_path = output_dir / f"{patient_id}_flair_3slice_compare.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _source_indices_for_last_target(num_sessions: int) -> list[int]:
    """
    Build 3 source-session indices for target = last session.

    Handling used here (training/inference-consistent 3-source requirement):
    - num_sessions > 3: use the most recent 3 historical sessions
    - num_sessions == 3: use the 2 historical sessions and duplicate the most recent
    - num_sessions < 3: duplicate the most recent available historical session
    """
    target = max(num_sessions - 1, 0)
    hist = list(range(target))
    if len(hist) >= 3:
        return hist[-3:]
    if len(hist) == 2:
        return [hist[0], hist[1], hist[1]]
    if len(hist) == 1:
        return [hist[0], hist[0], hist[0]]
    return [0, 0, 0]


def _load_patient_arrays(data_root: Path, patient_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.load(data_root / f"{patient_id}_image.npy")
    days = np.load(data_root / f"{patient_id}_days.npy")
    treat = np.load(data_root / f"{patient_id}_treatment.npy")
    t = int(len(days))
    if image.shape[0] % 4 != 0 or image.shape[0] // 4 != t:
        raise ValueError(f"{patient_id}: invalid image/session shape mismatch {image.shape}, T={t}")
    image = image.reshape(4, t, *image.shape[1:]).transpose(1, 0, 2, 3, 4)  # [T,4,H,W,D]
    image = image[:, :3, ...]  # keep T1,T1c,FLAIR
    return image.astype(np.float32), days.astype(np.float32), treat.astype(np.float32)


def _predict_slice_flair(
    model: Tadiff_model,
    image_t3d: np.ndarray,
    days: np.ndarray,
    treat: np.ndarray,
    seq_idx: Sequence[int],
    slice_idx: int,
    image_size: int,
    diffusion_steps: int,
    num_samples: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(seq_idx) != 4:
        raise ValueError(f"Expected 4 session indices [s1,s2,s3,target], got {seq_idx}")

    # [4,3,H,W] for selected slice
    seq_imgs = image_t3d[seq_idx, :, :, :, slice_idx]
    seq_imgs = _center_crop_or_pad_2d(seq_imgs, (image_size, image_size))
    gt_target = seq_imgs[-1]  # [3,H,W]

    seq = torch.from_numpy(seq_imgs).float().to(device)
    n_req = max(1, int(num_samples))
    # Avoid b==1 forward path in TaDiff_Net (has indexing edge cases with i_tg tensor).
    n_eff = max(2, n_req)
    x_t = seq.unsqueeze(0).repeat(n_eff, 1, 1, 1, 1)

    noise = torch.randn((n_req, 3, image_size, image_size), device=device)
    if n_eff > n_req:
        # Duplicate first requested sample to keep requested behavior unchanged.
        pad = noise[:1].repeat(n_eff - n_req, 1, 1, 1)
        noise = torch.cat([noise, pad], dim=0)
    x_t[:, -1, :, :, :] = noise
    x_t = x_t.reshape(n_eff, 12, image_size, image_size)

    # Build conditioning vectors explicitly to guarantee 4 entries.
    day_vals = [float(days[int(i)]) for i in seq_idx]
    tr_vals = [float(treat[int(i)]) for i in seq_idx]
    day_vec = torch.tensor(day_vals, dtype=torch.float32, device=device)
    tr_vec = torch.tensor(tr_vals, dtype=torch.float32, device=device)
    daysq = day_vec.unsqueeze(0).repeat(n_eff, 1)  # [n_eff,4]
    trq = tr_vec.unsqueeze(0).repeat(n_eff, 1)  # [n_eff,4]
    i_tg = torch.full((n_eff,), 3, dtype=torch.int8, device=device)

    diffusion = GaussianDiffusion(T=int(diffusion_steps), schedule="linear", device=device)
    pred_img, _ = diffusion.TaDiff_inverse(
        net=model,
        start_t=int(diffusion_steps),
        steps=int(diffusion_steps),
        x=x_t,
        intv=[daysq[:, i] for i in range(4)],
        treat_cond=[trq[:, i] for i in range(4)],
        i_tg=i_tg,
        device=device,
    )

    pred_avg = pred_img[:n_req].mean(dim=0).detach().cpu().numpy()  # [3,H,W]
    gt_np = gt_target.astype(np.float32)
    # FLAIR index is 2
    return pred_avg[2], gt_np[2]


def run_eval(
    ckpt_path: Path,
    data_root: Path,
    output_json: Path,
    diffusion_steps: int,
    num_samples: int,
    device: str,
) -> None:
    cfg = _load_eval_cfg_from_checkpoint(ckpt_path)
    patient_ids = list(cfg.testing_patient_ids)

    model = Tadiff_model.load_from_checkpoint(
        str(ckpt_path),
        config=cfg,
        strict=False,
        map_location=torch.device(device),
        weights_only=False,
    )
    model.to(device)
    model.eval()

    rows = []
    total_runs = 0
    vis_dir = output_json.parent / "visualizations"

    for patient_id in patient_ids:
        image_t3d, days, treat = _load_patient_arrays(data_root, patient_id)
        target_idx = image_t3d.shape[0] - 1
        src_idx = _source_indices_for_last_target(image_t3d.shape[0])
        seq_idx = src_idx + [target_idx]

        d = image_t3d.shape[-1]
        mid = d // 2
        slices = [mid - 1, mid, mid + 1]

        pred_stack_raw = []
        gt_stack_raw = []
        for z in slices:
            pred_flair, gt_flair = _predict_slice_flair(
                model=model,
                image_t3d=image_t3d,
                days=days,
                treat=treat,
                seq_idx=seq_idx,
                slice_idx=z,
                image_size=cfg.image_size,
                diffusion_steps=diffusion_steps,
                num_samples=num_samples,
                device=device,
            )
            pred_stack_raw.append(pred_flair)
            gt_stack_raw.append(gt_flair)
            total_runs += 1

        # Aggregate 3 slices as channels (3,H,W), normalize each tensor independently to [0,1].
        pred_chw_raw = np.stack(pred_stack_raw, axis=0).astype(np.float32)
        gt_chw_raw = np.stack(gt_stack_raw, axis=0).astype(np.float32)
        pred_chw_01, gt_chw_01 = _independent_minmax_to_01(pred_chw_raw, gt_chw_raw)

        # Save visualization for this patient: GT vs Pred for all 3 slices.
        _save_patient_visualization(
            patient_id=patient_id,
            slices=slices,
            pred_stack=list(pred_chw_01),
            gt_stack=list(gt_chw_01),
            output_dir=vis_dir,
        )

        # Compute one SSIM on the stacked 3-channel input.
        ssim_val = _ssim_tadiff_from_numpy_chw(pred_chw_01, gt_chw_01)

        rows.append(
            {
                "patient_id": patient_id,
                "num_sessions": int(image_t3d.shape[0]),
                "source_session_indices": src_idx,
                "target_session_index": int(target_idx),
                "slice_indices": slices,
                "ssim_flair_3slice": float(ssim_val),
            }
        )
        print(
            f"{patient_id}: sessions={image_t3d.shape[0]} "
            f"src={src_idx} tgt={target_idx} slices={slices} "
            f"SSIM={ssim_val:.6f}"
        )

    mean_ssim = float(np.mean([r["ssim_flair_3slice"] for r in rows])) if rows else float("nan")
    std_error = float(np.std([r["ssim_flair_3slice"] for r in rows]) / np.sqrt(len(rows)))
    out = {
        "num_patients": len(rows),
        "num_inference_runs": total_runs,
        "diffusion_steps": int(diffusion_steps),
        "num_samples_per_slice": int(num_samples),
        "mean_ssim_flair_3slice": mean_ssim,
        "patients": rows,
        "std_error": std_error,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {output_json}")
    print(f"Patients: {len(rows)} | Inference runs: {total_runs} | Mean SSIM: {mean_ssim:.6f}")
    print(f"Standard error: {std_error:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TaDiff 3-slice FLAIR SSIM on MU-Glioma-Post.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained .ckpt file")
    parser.add_argument("--data_root", type=str, default="./mu_glioma_post_output")
    parser.add_argument("--output_json", type=str, default="./eval_t2f_generation/ssim_results.json")
    parser.add_argument("--diffusion_steps", type=int, default=600)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        ckpt_path=Path(args.ckpt_path),
        data_root=Path(args.data_root),
        output_json=Path(args.output_json),
        diffusion_steps=args.diffusion_steps,
        num_samples=args.num_samples,
        device=args.device,
    )
