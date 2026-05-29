"""
Evaluate TaDiff FLAIR generation RVD on MU-Glioma-Post test patients.

Protocol requested by user:
- Patients: use testing_patient_ids from train_mu_glioma_post_config.py
- Data source: mu_glioma_post_output/*.npy
- Target session: last session of each patient
- Slices: middle three slices [D//2 - 1, D//2, D//2 + 1]
- Run model 3 times per patient (one per slice), total 20*3 = 60 runs
- Keep only FLAIR modality from prediction and GT
- Stack 3 FLAIR slices as channels and compute per-slice RVD on tumor masks

RVD pipeline:
1) Generate the same 3-slice prediction triplet as psnr.py.
2) Convert pred/GT to [0,1] using independent min-max (same as psnr.py).
3) Segment each channel independently.
4) Compute per-slice RVD: (V_pred - V_gt) / V_gt.
5) Report per-slice RVD, patient average, and dataset average.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage as ndi

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


def _independent_minmax_to_01(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Independent min-max normalization to [0,1] for pred and GT separately.

    This matches psnr.py normalization behavior.
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


def _prepare_slice_for_segmentation(slice_01: np.ndarray) -> np.ndarray:
    x = np.asarray(slice_01, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)


def _compute_rvd(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-8) -> float:
    vp = float(np.count_nonzero(pred_mask))
    vg = float(np.count_nonzero(gt_mask))
    if vg <= eps:
        return 0.0 if vp <= eps else float("nan")
    return (vp - vg) / vg


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
    pred_01: np.ndarray,
    gt_01: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save one PNG per patient.
    Layout: 4 rows x 3 cols
      row 1: GT intensity
      row 2: Pred intensity
      row 3: GT segmentation mask
      row 4: Pred segmentation mask
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
    for col in range(3):
        axes[0, col].imshow(gt_01[col], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"GT ch={col}")
        axes[0, col].axis("off")

        axes[1, col].imshow(pred_01[col], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"Pred ch={col}")
        axes[1, col].axis("off")

        axes[2, col].imshow(gt_mask[col], cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_title(f"GT mask ch={col}")
        axes[2, col].axis("off")

        axes[3, col].imshow(pred_mask[col], cmap="gray", vmin=0, vmax=1)
        axes[3, col].set_title(f"Pred mask ch={col}")
        axes[3, col].axis("off")

    fig.suptitle(f"{patient_id} | RVD intensity + mask comparison", fontsize=12)
    save_path = output_dir / f"{patient_id}_rvd_compare.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


@dataclass
class SegmenterConfig:
    backend: str = "flair_adaptive"
    threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SliceSegmenter:
    def segment(self, slice_01: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ThresholdSliceSegmenter(SliceSegmenter):
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def segment(self, slice_01: np.ndarray) -> np.ndarray:
        x = _prepare_slice_for_segmentation(slice_01)
        return (x >= self.threshold).astype(np.uint8)


class TorchHubLggUNetSegmenter(SliceSegmenter):
    """
    Pretrained model:
      torch.hub.load("mateuszbuda/brain-segmentation-pytorch", "unet", pretrained=True)
    The published model expects 3-channel MRI input; for single-slice 2D segmentation we
    repeat one normalized slice to 3 channels.
    """

    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def segment(self, slice_01: np.ndarray) -> np.ndarray:
        x = _prepare_slice_for_segmentation(slice_01)
        x3 = np.stack([x, x, x], axis=0)
        inp = torch.from_numpy(x3).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            prob = self.model(inp).sigmoid().squeeze(0).squeeze(0).cpu().numpy()
        return (prob >= self.threshold).astype(np.uint8)


class FlairAdaptiveSliceSegmenter(SliceSegmenter):
    """
    Heuristic 2D FLAIR tumor segmentation for a single normalized slice:
    - robustly estimate brain foreground
    - detect hyperintense tumor candidates via z-score and top-percentile
    - clean mask with morphology and small-component removal
    """

    def __init__(
        self,
        *,
        z_score: float = 1.0,
        top_percentile: float = 97.0,
        min_area_ratio: float = 3e-4,
    ):
        self.z_score = float(z_score)
        self.top_percentile = float(top_percentile)
        self.min_area_ratio = float(min_area_ratio)

    def _brain_mask(self, x: np.ndarray) -> np.ndarray:
        # Keep non-background tissue; robust against tiny normalization shifts.
        p20 = float(np.percentile(x, 20.0))
        thr = max(0.02, p20 * 0.5)
        return x > thr

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return mask
        labeled, n_comp = ndi.label(mask)
        if n_comp == 0:
            return np.zeros_like(mask, dtype=bool)
        min_area = max(8, int(mask.size * self.min_area_ratio))
        sizes = np.bincount(labeled.ravel())
        keep = sizes >= min_area
        keep[0] = False
        return keep[labeled]

    def segment(self, slice_01: np.ndarray) -> np.ndarray:
        x = _prepare_slice_for_segmentation(slice_01)
        brain = self._brain_mask(x)
        vals = x[brain]
        if vals.size < 32:
            return np.zeros_like(x, dtype=np.uint8)

        mu = float(vals.mean())
        sigma = float(vals.std())
        z_thr = mu + self.z_score * (sigma + 1e-8)
        pct_thr = float(np.percentile(vals, self.top_percentile))
        signal_thr = max(z_thr, pct_thr)

        raw = (x >= signal_thr) & brain
        cleaned = ndi.binary_closing(raw, structure=np.ones((3, 3), dtype=bool))
        cleaned = ndi.binary_fill_holes(cleaned)
        cleaned = self._remove_small_components(cleaned)
        return cleaned.astype(np.uint8)


def _build_segmenter(cfg: SegmenterConfig) -> SliceSegmenter:
    if cfg.backend == "flair_adaptive":
        return FlairAdaptiveSliceSegmenter()
    if cfg.backend == "threshold":
        return ThresholdSliceSegmenter(threshold=cfg.threshold)
    if cfg.backend == "torchhub_lgg_unet":
        return TorchHubLggUNetSegmenter(threshold=cfg.threshold, device=cfg.device)
    raise ValueError(
        f"Unsupported segmentation backend: {cfg.backend}. "
        f"Choose from: 'flair_adaptive', 'torchhub_lgg_unet', 'threshold'."
    )


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
    seg_backend: str,
    seg_threshold: float,
    print_mask_stats: bool,
    save_visualization: bool,
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

    segmenter = _build_segmenter(
        SegmenterConfig(backend=seg_backend, threshold=seg_threshold, device=device)
    )

    rows = []
    total_runs = 0
    vis_dir = output_json.parent / "rvd_visualizations"

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

        rvd_values = []
        volume_pairs = []
        pred_masks = []
        gt_masks = []
        for c in range(pred_chw_01.shape[0]):
            pred_mask = segmenter.segment(pred_chw_01[c])
            gt_mask = segmenter.segment(gt_chw_01[c])
            pred_masks.append(pred_mask)
            gt_masks.append(gt_mask)
            vp = int(np.count_nonzero(pred_mask))
            vg = int(np.count_nonzero(gt_mask))
            volume_pairs.append((vp, vg))
            rvd_values.append(_compute_rvd(pred_mask, gt_mask))

        rvd_arr = np.asarray(rvd_values, dtype=np.float64)
        if np.all(np.isnan(rvd_arr)):
            print(f"[skip] {patient_id}: all 3 slices have empty GT tumor mask")
            continue

        patient_avg = float(np.nanmean(rvd_arr))
        rows.append(
            {
                "patient_id": patient_id,
                "num_sessions": int(image_t3d.shape[0]),
                "source_session_indices": src_idx,
                "target_session_index": int(target_idx),
                "slice_indices": slices,
                "rvd_flair_3slice_per_slice": [float(v) if np.isfinite(v) else None for v in rvd_arr.tolist()],
                "rvd_flair_3slice_patient_avg": patient_avg,
            }
        )

        if print_mask_stats:
            vol_text = ", ".join(f"(pred={vp}, gt={vg})" for vp, vg in volume_pairs)
            print(f"{patient_id}: mask volumes per-slice = [{vol_text}]")

        if save_visualization:
            _save_patient_visualization(
                patient_id=patient_id,
                pred_01=pred_chw_01,
                gt_01=gt_chw_01,
                pred_mask=np.stack(pred_masks, axis=0),
                gt_mask=np.stack(gt_masks, axis=0),
                output_dir=vis_dir,
            )

        rvd_text = ", ".join(f"{v:.6f}" if np.isfinite(v) else "nan" for v in rvd_arr.tolist())
        print(
            f"{patient_id}: sessions={image_t3d.shape[0]} "
            f"src={src_idx} tgt={target_idx} slices={slices} "
            f"RVD per-slice=[{rvd_text}] patient_avg={patient_avg:.6f}"
        )

    mean_rvd = (
        float(np.mean([r["rvd_flair_3slice_patient_avg"] for r in rows])) if rows else float("nan")
    )
    std_error = float(np.std([r["rvd_flair_3slice_patient_avg"] for r in rows]) / np.sqrt(len(rows)))
    out = {
        "num_patients": len(rows),
        "num_inference_runs": total_runs,
        "diffusion_steps": int(diffusion_steps),
        "num_samples_per_slice": int(num_samples),
        "seg_backend": seg_backend,
        "seg_threshold": float(seg_threshold),
        "mean_rvd_flair_3slice_patient_avg": mean_rvd,
        "patients": rows,
        "std_error": std_error,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {output_json}")
    print(f"Patients: {len(rows)} | Inference runs: {total_runs} | Mean RVD: {mean_rvd:.6f}")
    print(f"Standard error: {std_error:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TaDiff 3-slice FLAIR RVD on MU-Glioma-Post.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained .ckpt file")
    parser.add_argument("--data_root", type=str, default="./mu_glioma_post_output")
    parser.add_argument("--output_json", type=str, default="./eval_t2f_generation/rvd_results.json")
    parser.add_argument("--diffusion_steps", type=int, default=600)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seg_backend", type=str, default=os.getenv("RVD_SEG_BACKEND", "flair_adaptive"))
    parser.add_argument(
        "--seg_threshold",
        type=float,
        default=float(os.getenv("RVD_SEG_THRESHOLD", "0.5")),
        help="Used by threshold/torchhub_lgg_unet backends.",
    )
    parser.add_argument(
        "--print_mask_stats",
        action="store_true",
        default=os.getenv("RVD_PRINT_MASK_STATS", "1") != "0",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        default=os.getenv("RVD_SAVE_VIS", "1") != "0",
    )
    parser.add_argument(
        "--no_save_visualization",
        action="store_false",
        dest="save_visualization",
    )
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
        seg_backend=args.seg_backend,
        seg_threshold=args.seg_threshold,
        print_mask_stats=args.print_mask_stats,
        save_visualization=args.save_visualization,
    )
