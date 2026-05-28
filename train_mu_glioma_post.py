from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from config.train_mu_glioma_post_config import TrainMuGliomaPostConfig
from eval_t2f_generation.ssim import (
    _independent_minmax_to_01,
    _load_patient_arrays,
    _predict_slice_flair,
    _source_indices_for_last_target,
)
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


class PatientStudy:
    def __init__(self, patient_id: str, image: np.ndarray, label: np.ndarray, days: np.ndarray, treatment: np.ndarray):
        self.patient_id = patient_id
        self.image = image  # [T,4,H,W,D]
        self.label = label  # [T,H,W,D]
        self.days = days    # [T]
        self.treatment = treatment  # [T]


def _load_study(data_root: Path, patient_id: str) -> PatientStudy:
    image = np.load(data_root / f"{patient_id}_image.npy")
    label = np.load(data_root / f"{patient_id}_label.npy")
    days = np.load(data_root / f"{patient_id}_days.npy")
    treatment = np.load(data_root / f"{patient_id}_treatment.npy")

    t = int(len(days))
    if image.shape[0] % 4 != 0 or image.shape[0] // 4 != t:
        raise ValueError(f"{patient_id}: image shape/session mismatch: image={image.shape}, T={t}")
    if label.shape[0] != t or treatment.shape[0] != t:
        raise ValueError(f"{patient_id}: label/days/treatment mismatch")

    # SAILOR-style ordering: modality-major then time.
    image_t4 = image.reshape(4, t, *image.shape[1:]).transpose(1, 0, 2, 3, 4)  # [T,4,H,W,D]
    return PatientStudy(
        patient_id=patient_id,
        image=image_t4.astype(np.float32),
        label=label.astype(np.float32),
        days=days.astype(np.float32),
        treatment=treatment.astype(np.float32),
    )


def _pick_sources(pool: Sequence[int], n_sources: int = 3) -> List[int]:
    if len(pool) == 0:
        return [0] * n_sources
    if len(pool) >= n_sources:
        out = random.sample(list(pool), n_sources)
    else:
        out = [random.choice(pool) for _ in range(n_sources)]
    out.sort()
    return out


def _pick_indices(num_sessions: int, future_prob: float, middle_prob: float, past_prob: float) -> List[int]:
    if num_sessions <= 1:
        return [0, 0, 0, 0]

    r = random.random()
    p1 = future_prob
    p2 = future_prob + middle_prob

    if r < p1:
        # Future target: sources from historical sessions.
        target = random.randint(1, num_sessions - 1)
        sources = _pick_sources(list(range(0, target)), 3)
    elif r < p2 and num_sessions >= 3:
        # Middle target: can use both past/future sessions.
        target = random.randint(1, num_sessions - 2)
        pool = [i for i in range(num_sessions) if i != target]
        sources = _pick_sources(pool, 3)
    else:
        # Past target: sources from later sessions.
        target = random.randint(0, num_sessions - 2)
        sources = _pick_sources(list(range(target + 1, num_sessions)), 3)

    return sources + [target]


class LongitudinalSliceDataset(Dataset):
    def __init__(self, studies: List[PatientStudy], cfg: TrainMuGliomaPostConfig, samples_per_epoch: int):
        self.studies = studies
        self.cfg = cfg
        self.samples_per_epoch = max(1, int(samples_per_epoch))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        study = random.choice(self.studies)
        t = study.label.shape[0]
        sess_idx = _pick_indices(
            num_sessions=t,
            future_prob=self.cfg.future_prob,
            middle_prob=self.cfg.middle_prob,
            past_prob=self.cfg.past_prob,
        )

        d = study.image.shape[-1]
        z = random.randint(0, d - 1)

        # Image [4,3,H,W], label [4,H,W]
        img = study.image[sess_idx, :3, :, :, z]
        lbl = study.label[sess_idx, :, :, z]

        img = _center_crop_or_pad_2d(img, (self.cfg.image_size, self.cfg.image_size))
        lbl = _center_crop_or_pad_2d(lbl, (self.cfg.image_size, self.cfg.image_size))

        # Binary tumor mask for segmentation branch.
        lbl = (lbl > 0).astype(np.float32)

        days = study.days[sess_idx].astype(np.float32)
        tr = study.treatment[sess_idx].astype(np.float32)

        return {
            "image": torch.from_numpy(img).float(),         # [4,3,H,W]
            "label": torch.from_numpy(lbl).float(),         # [4,H,W]
            "days": torch.from_numpy(days).float(),         # [4]
            "treatments": torch.from_numpy(tr).float(),     # [4]
        }


class MuGliomaDataModule(pl.LightningDataModule):
    def __init__(self, cfg: TrainMuGliomaPostConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds: LongitudinalSliceDataset | None = None
        self.val_ds: LongitudinalSliceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        data_root = Path(self.cfg.data_root)
        image_files = sorted(data_root.glob("*_image.npy"))
        patient_ids = [p.name.replace("_image.npy", "") for p in image_files]
        if not patient_ids:
            raise RuntimeError(f"No *_image.npy found in {data_root}")

        test_set = set(self.cfg.testing_patient_ids)
        train_ids = [pid for pid in patient_ids if pid not in test_set]
        if not train_ids:
            raise RuntimeError("No training patients left after applying testing_patient_ids split.")

        train_studies = [_load_study(data_root, pid) for pid in train_ids]

        # Keep epoch size stable regardless of patient count.
        samples_per_epoch = max(2000, len(train_studies) * 200)
        self.train_ds = LongitudinalSliceDataset(train_studies, self.cfg, samples_per_epoch=samples_per_epoch)

        if self.cfg.use_validation:
            # Optional tiny validation by reusing training studies with fewer samples.
            self.val_ds = LongitudinalSliceDataset(train_studies, self.cfg, samples_per_epoch=max(128, len(train_studies) * 20))
        else:
            self.val_ds = None

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            return DataLoader([])
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=max(1, self.cfg.num_workers // 2),
            pin_memory=True,
            drop_last=False,
        )


class TadiffPaperModel(Tadiff_model):
    """
    Keep original Tadiff_model loss/forward behavior, but log train loss at both
    step and epoch for easier monitoring.
    """

    def training_step(self, batch, batch_idx):
        loss, mse, dice_seg = self.get_loss(batch, mode="train")
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mse", mse, sync_dist=True, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_dice", dice_seg, sync_dist=True, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "mse": mse, "dice_seg": dice_seg}


class PredictionPreviewCallback(pl.Callback):
    """
    Periodically run one inference preview for a fixed patient and log to TensorBoard.
    """

    def __init__(self, cfg: TrainMuGliomaPostConfig):
        super().__init__()
        self.cfg = cfg
        self._loaded = False
        self._image_t3d: np.ndarray | None = None
        self._days: np.ndarray | None = None
        self._treat: np.ndarray | None = None
        self._seq_idx: List[int] | None = None
        self._slice_idx: int | None = None

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        data_root = Path(self.cfg.data_root)
        pid = self.cfg.preview_patient_id
        image_t3d, days, treat = _load_patient_arrays(data_root, pid)
        target_idx = image_t3d.shape[0] - 1
        src_idx = _source_indices_for_last_target(image_t3d.shape[0])
        seq_idx = src_idx + [target_idx]
        d = image_t3d.shape[-1]
        mid = d // 2
        self._image_t3d = image_t3d
        self._days = days
        self._treat = treat
        self._seq_idx = seq_idx
        self._slice_idx = mid
        self._loaded = True

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.cfg.preview_interval_steps <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0 or (step % int(self.cfg.preview_interval_steps)) != 0:
            return
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return

        try:
            self._lazy_load()
            assert self._image_t3d is not None
            assert self._days is not None
            assert self._treat is not None
            assert self._seq_idx is not None
            assert self._slice_idx is not None

            pred_flair, gt_flair = _predict_slice_flair(
                model=pl_module,
                image_t3d=self._image_t3d,
                days=self._days,
                treat=self._treat,
                seq_idx=self._seq_idx,
                slice_idx=self._slice_idx,
                image_size=self.cfg.image_size,
                diffusion_steps=self.cfg.max_T,
                num_samples=1,
                device=str(pl_module.device),
            )
            pred_01, gt_01 = _independent_minmax_to_01(
                pred_flair.astype(np.float32), gt_flair.astype(np.float32)
            )
            diff_01 = np.abs(pred_01 - gt_01).astype(np.float32)
            panel = np.concatenate([gt_01, pred_01, diff_01], axis=1)  # [H, 3W]
            panel = np.clip(panel, 0.0, 1.0)
            panel_t = torch.from_numpy(panel).unsqueeze(0)  # [1,H,W]

            trainer.logger.experiment.add_image(
                f"preview/{self.cfg.preview_patient_id}_flair_gt_pred_diff",
                panel_t,
                global_step=step,
            )
        except Exception as e:
            # Keep training running even if preview fails.
            print(f"[PreviewCallback] skipped at step {step}: {e}")


def _make_trainer(cfg: TrainMuGliomaPostConfig) -> pl.Trainer:
    logger = TensorBoardLogger(save_dir=cfg.logdir, name="", default_hp_metric=False)

    monitor_name = "val_loss" if cfg.use_validation else "train_loss_epoch"
    monitor_mode = "min"
    ckpt_cb = ModelCheckpoint(
        dirpath=str(Path(cfg.logdir) / "checkpoints"),
        save_top_k=cfg.ckpt_save_top_k,
        save_last=cfg.ckpt_save_last,
        monitor=monitor_name,
        mode=monitor_mode,
        filename=cfg.ckpt_filename,
        every_n_epochs=cfg.val_interval_epoch if cfg.use_validation else None,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[ckpt_cb, PredictionPreviewCallback(cfg)],
        precision=cfg.precision,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
        max_epochs=cfg.max_epochs if cfg.max_epochs > 0 else -1,
        gradient_clip_val=cfg.grad_clip,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        log_every_n_steps=cfg.log_interval,
        check_val_every_n_epoch=cfg.val_interval_epoch,
        num_sanity_val_steps=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TaDiff on MU-Glioma-Post.")
    parser.add_argument("--data_root", type=str, default=None, help="Override config data_root")
    parser.add_argument("--logdir", type=str, default=None, help="Override config logdir")
    parser.add_argument("--max_steps", type=int, default=None, help="Override config max_steps")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override config max_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override config num_workers")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainMuGliomaPostConfig()

    # Optional CLI overrides
    for key in ("data_root", "logdir", "max_steps", "max_epochs", "batch_size", "num_workers", "lr", "seed"):
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)

    pl.seed_everything(cfg.seed, workers=True)

    dm = MuGliomaDataModule(cfg)
    model = TadiffPaperModel(config=cfg)

    trainer = _make_trainer(cfg)
    print("Training config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

