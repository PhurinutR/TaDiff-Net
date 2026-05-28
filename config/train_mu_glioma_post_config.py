from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainMuGliomaPostConfig:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_root: str = "./mu_glioma_post_output"
    npz_keys: Tuple[str, ...] = ("image", "label", "days", "treatment")
    image_size: int = 192

    # Fixed test split requested by user.
    testing_patient_ids: Tuple[str, ...] = (
        "PatientID_0036",
        "PatientID_0201",
        "PatientID_0030",
        "PatientID_0132",
        "PatientID_0198",
        "PatientID_0019",
        "PatientID_0204",
        "PatientID_0272",
        "PatientID_0260",
        "PatientID_0252",
        "PatientID_0125",
        "PatientID_0255",
        "PatientID_0186",
        "PatientID_0026",
        "PatientID_0095",
        "PatientID_0208",
        "PatientID_0267",
        "PatientID_0051",
        "PatientID_0189",
        "PatientID_0099",
    )
    use_validation: bool = False
    val_fraction: float = 0.1

    # ------------------------------------------------------------------
    # Paper-style sampling strategy
    # ------------------------------------------------------------------
    future_prob: float = 0.5
    middle_prob: float = 0.3
    past_prob: float = 0.2

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    network: str = "TaDiff_Net"
    in_channels: int = 13
    out_channels: int = 7
    model_channels: int = 64
    num_res_blocks: int = 2
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)
    attention_resolutions: Tuple[int, ...] = (8, 4)
    num_heads: int = 8
    num_classes: int = 81

    # Diffusion / loss
    max_T: int = 600
    ddpm_schedule: str = "linear"
    aux_loss_w: float = 0.01

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    opt: str = "adamw"
    lr: float = 2.5e-4
    weight_decay: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 5_000_000
    max_epochs: int = -1
    batch_size: int = 32
    num_workers: int = 8
    grad_clip: float = 1.5
    accumulate_grad_batches: int = 2
    precision: int = 32

    # ------------------------------------------------------------------
    # Logging / checkpoints / preview
    # ------------------------------------------------------------------
    seed: int = 114514
    logdir: str = "./lightning_logs/train_mu_glioma_post"
    log_interval: int = 1
    val_interval_epoch: int = 1

    ckpt_save_top_k: int = 3
    ckpt_save_last: bool = True
    ckpt_monitor: str = "train_loss_epoch"
    ckpt_mode: str = "min"
    ckpt_filename: str = "ckpt-{epoch}-{step}-{train_loss_epoch:.6f}"

    preview_patient_id: str = "PatientID_0036"
    preview_interval_steps: int = 1000
    preview_slice: str = "middle"

