"""Loop de treinamento PyTorch com suporte a CUDA/CPU, AdamW e Cosine Decay.

Conforme especificação:
- Suporte a GPU CUDA (NVIDIA RTX PRO 1000) e CPU.
- Otimizador AdamW com decaimento por cosseno (CosineAnnealingLR).
- Validação contínua calculando MAE residual de resposta, erro em curvas identidade (teste de não-agressão)
  e métricas M1 a M4 da spec 08-eq-neural.md.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from neural_eq.data import (
    SyntheticEqDataset,
    create_consistency_evaluation_cases,
)
from neural_eq.export import export_neural_eq_onnx, validate_onnx_parity, verify_tract_compatibility
from neural_eq.loss import NeuralEqLoss
from neural_eq.metrics import evaluate_eq_metrics
from neural_eq.model import NeuralEqConfig, NeuralEqModel

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    train_samples: int = 8000
    val_samples: int = 1000
    ood_samples: int = 500
    identity_samples: int = 500
    checkpoint_dir: str = "runs/eq/checkpoints"
    export_onnx_path: str | None = "runs/eq/neural_eq.onnx"


def train_neural_eq(config: TrainConfig | None = None) -> tuple[NeuralEqModel, dict[str, Any]]:
    """Treina o modelo Neural EQ e executa validação contínua."""
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    logger.info("Iniciando treinamento no dispositivo: %s", device)

    # Inicialização de datasets e dataloaders
    train_ds = SyntheticEqDataset(
        num_samples=cfg.train_samples,
        identity_ratio=0.25,
        is_ood=False,
        seed=cfg.seed,
    )
    val_ds = SyntheticEqDataset(
        num_samples=cfg.val_samples,
        identity_ratio=0.25,
        is_ood=False,
        seed=cfg.seed + 1,
    )
    ood_ds = SyntheticEqDataset(
        num_samples=cfg.ood_samples,
        is_ood=True,
        seed=cfg.seed + 2,
    )
    ident_ds = SyntheticEqDataset(
        num_samples=cfg.identity_samples,
        identity_ratio=1.0,
        seed=cfg.seed + 3,
    )
    consistency_cases = create_consistency_evaluation_cases(
        num_channels=30,
        variations_per_channel=5,
        seed=cfg.seed + 4,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    ood_loader = DataLoader(ood_ds, batch_size=cfg.batch_size, shuffle=False)
    ident_loader = DataLoader(ident_ds, batch_size=cfg.batch_size, shuffle=False)

    model = NeuralEqModel(NeuralEqConfig()).to(device)
    loss_fn = NeuralEqLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    ckpt_path = Path(cfg.checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    best_m1_median = float("inf")
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for feat, curve, is_ident in train_loader:
            feat = feat.to(device)
            curve = curve.to(device)
            is_ident = is_ident.to(device)

            optimizer.zero_grad()
            pred_gains = model(feat)
            loss_dict = loss_fn(pred_gains, curve, is_identity=is_ident)
            loss = loss_dict["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(total_batches, 1)

        # Validação contínua a cada época com cálculo das métricas M1 a M4
        metrics = evaluate_eq_metrics(
            model=model,
            val_loader=val_loader,
            ood_loader=ood_loader,
            ident_loader=ident_loader,
            consistency_cases=consistency_cases,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            **metrics,
        }
        history.append(epoch_record)

        logger.info(
            "Época [%02d/%02d] Loss: %.4f | M1 Resid Mediana: %.2f dB (Pass: %s) | "
            "M3 Ident Mean: %.3f dB (Pass: %s) | M4 Std: %.3f dB",
            epoch,
            cfg.epochs,
            avg_train_loss,
            metrics["m1_residual_median_db"],
            metrics["m1_pass"],
            metrics["m3_ident_mean_abs_g_db"],
            metrics["m3_pass"],
            metrics["m4_consistency_mean_std_db"],
        )

        # Prioriza checkpoint com m1_pass e m3_pass e menor mediana residual M1
        passes = metrics.get("m1_pass", False) and metrics.get("m3_pass", False)
        best_passed = best_metrics.get("m1_pass", False) and best_metrics.get("m3_pass", False)
        is_better = (passes and not best_passed) or (
            passes == best_passed and metrics["m1_residual_median_db"] < best_m1_median
        )
        if is_better:
            best_m1_median = metrics["m1_residual_median_db"]
            best_metrics = metrics
            best_file = ckpt_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": asdict(cfg),
                },
                best_file,
            )

    # Carrega melhor modelo para exportação final
    best_file = ckpt_path / "best_model.pt"
    if best_file.is_file():
        ckpt = torch.load(best_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Exportação e validações ONNX se configurado
    if cfg.export_onnx_path:
        export_path = Path(cfg.export_onnx_path)
        export_neural_eq_onnx(model, export_path)
        parity_diff = validate_onnx_parity(model, export_path)
        tract_ok = verify_tract_compatibility(export_path)
        best_metrics["onnx_path"] = str(export_path)
        best_metrics["onnx_parity_diff_db"] = parity_diff
        best_metrics["tract_compatible"] = tract_ok
        logger.info(
            "ONNX exportado em %s com paridade max_diff=%.2e dB (Tract: %s)",
            export_path,
            parity_diff,
            tract_ok,
        )

    results = {
        "best_metrics": best_metrics,
        "history": history,
        "best_checkpoint": str(best_file),
    }
    return model, results
