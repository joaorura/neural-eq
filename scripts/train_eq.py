#!/usr/bin/env python3
"""Script CLI unificado para treinamento e exportação do Neural EQ.

Uso:
    python scripts/train_eq.py --epochs 15 --batch-size 64 --device cuda --export-onnx runs/eq/neural_eq.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Adiciona src ao path de importação
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_eq.train import TrainConfig, train_neural_eq  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_eq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento e exportação do Neural EQ (Clearcore Subprojeto B)")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas de treinamento")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do lote")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado inicial")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay para o AdamW")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo de execução ('cuda' ou 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Semente pseudo-aleatória")
    parser.add_argument("--train-samples", type=int, default=8000, help="Quantidade de amostras de treino por época")
    parser.add_argument("--val-samples", type=int, default=1000, help="Quantidade de amostras de validação in-distribution")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/eq/checkpoints", help="Diretório para checkpoints")
    parser.add_argument(
        "--export-onnx",
        type=str,
        default="runs/eq/neural_eq.onnx",
        help="Caminho para exportação do modelo ONNX final (opcional)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        checkpoint_dir=args.checkpoint_dir,
        export_onnx_path=args.export_onnx,
    )

    logger.info("Configurações de treino: %s", cfg)
    _, results = train_neural_eq(cfg)

    metrics = results["best_metrics"]
    logger.info("=== RESULTADOS FINAIS DO NEURAL EQ ===")
    logger.info("M1 Resid Mediana: %.3f dB (Pass: %s)", metrics.get("m1_residual_median_db", -1), metrics.get("m1_pass"))
    logger.info("M1 Resid P90:     %.3f dB", metrics.get("m1_residual_p90_db", -1))
    logger.info("M2 OOD Mediana:   %.3f dB (Pass: %s)", metrics.get("m2_ood_median_db", -1), metrics.get("m2_pass"))
    logger.info("M3 Ident Mean |g|: %.3f dB (Pass: %s)", metrics.get("m3_ident_mean_abs_g_db", -1), metrics.get("m3_pass"))
    logger.info("M3 Ident P95 |g|:  %.3f dB", metrics.get("m3_ident_p95_abs_g_db", -1))
    logger.info("M4 Desvio Médio:  %.3f dB (Pass: %s)", metrics.get("m4_consistency_mean_std_db", -1), metrics.get("m4_pass"))
    logger.info("Todas as métricas aprovadas: %s", metrics.get("all_metrics_pass"))
    logger.info("Melhor checkpoint: %s", results.get("best_checkpoint"))
    if args.export_onnx:
        logger.info("ONNX salvo em: %s", metrics.get("onnx_path"))
        logger.info("Paridade máxima vs ORT: %.2e dB", metrics.get("onnx_parity_diff_db", -1))

    # Grava relatório json resumido ao lado do checkpoint
    summary_path = Path(args.checkpoint_dir) / "metrics_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Sumário salvo em %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
