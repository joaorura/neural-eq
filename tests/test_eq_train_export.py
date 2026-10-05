"""Testes unitários para o treinamento, validação e exportação ONNX do Neural EQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from neural_eq.constants import (
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    NUM_ERB_BANDS,
    TOTAL_FFT_BINS,
    build_dct_basis,
    build_interpolation_matrix,
    compute_band_centers,
)
from neural_eq.export import (
    export_neural_eq_onnx,
    validate_onnx_parity,
    verify_tract_compatibility,
)
from neural_eq.loss import NeuralEqLoss
from neural_eq.model import NeuralEqConfig, NeuralEqModel
from neural_eq.train import TrainConfig, train_neural_eq


def test_constants_and_interpolation():
    centers = compute_band_centers()
    assert len(centers) == NUM_ERB_BANDS
    assert centers[0] < centers[1] < centers[-1]

    m = build_interpolation_matrix()
    assert m.shape == (TOTAL_FFT_BINS, NUM_ERB_BANDS)

    # Soma de cada linha de M deve ser 1.0 (propriedade de interpolação linear afim)
    row_sums = m.sum(dim=-1).numpy()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    # Ganhos planos de 6.0 dB devem gerar resposta exatamente de 6.0 dB em todos os bins
    flat_gains = torch.full((1, NUM_ERB_BANDS), 6.0)
    flat_response = flat_gains @ m.T
    np.testing.assert_allclose(flat_response.numpy(), 6.0, atol=1e-5)

    # Base DCT ortonormal (32, 8)
    basis = build_dct_basis(NUM_ERB_BANDS, 8)
    assert basis.shape == (NUM_ERB_BANDS, 8)
    gram = basis.T @ basis
    np.testing.assert_allclose(gram.numpy(), np.eye(8), atol=1e-5)


def test_neural_eq_model_forward():
    model = NeuralEqModel(NeuralEqConfig())
    model.eval()

    # Formato [batch, 32, 3]
    x = torch.randn(4, NUM_ERB_BANDS, 3)
    out = model(x)
    assert out.shape == (4, NUM_ERB_BANDS)

    # Formato alternativo [batch, 96]
    x_flat = torch.randn(4, 96)
    out_flat = model(x_flat)
    assert out_flat.shape == (4, NUM_ERB_BANDS)

    # Entradas extremas devem ser estritamente limitadas em [-6.0, +12.0]
    extreme_x = torch.full((2, NUM_ERB_BANDS, 3), 100.0)
    out_extreme = model(extreme_x)
    assert torch.all(out_extreme >= GAIN_MIN_DB)
    assert torch.all(out_extreme <= GAIN_MAX_DB)

    extreme_neg = torch.full((2, NUM_ERB_BANDS, 3), -100.0)
    out_neg = model(extreme_neg)
    assert torch.all(out_neg >= GAIN_MIN_DB)
    assert torch.all(out_neg <= GAIN_MAX_DB)


def test_neural_eq_loss():
    loss_fn = NeuralEqLoss()
    pred_gains = torch.randn(4, NUM_ERB_BANDS, requires_grad=True)
    curve = torch.randn(4, TOTAL_FFT_BINS)
    is_ident = torch.tensor([True, False, False, True])

    loss_dict = loss_fn(pred_gains, curve, is_identity=is_ident)
    assert "loss_total" in loss_dict
    assert "loss_residual" in loss_dict
    assert "loss_identity" in loss_dict

    total_loss = loss_dict["loss_total"]
    total_loss.backward()
    assert pred_gains.grad is not None
    assert torch.isfinite(pred_gains.grad).all()


def test_train_smoke_and_export(tmp_path: Path):
    """Executa um ciclo rápido de 2 épocas de treinamento e valida o ONNX exportado."""
    ckpt_dir = tmp_path / "checkpoints"
    onnx_path = tmp_path / "neural_eq_smoke.onnx"

    cfg = TrainConfig(
        epochs=2,
        batch_size=16,
        lr=2e-3,
        device="cpu",  # CPU para execução garantida e rápida no teste
        seed=123,
        train_samples=64,
        val_samples=32,
        ood_samples=16,
        identity_samples=16,
        checkpoint_dir=str(ckpt_dir),
        export_onnx_path=str(onnx_path),
    )

    model, results = train_neural_eq(cfg)

    # Verifica se histórico de 2 épocas foi produzido
    assert len(results["history"]) == 2
    best_file = Path(results["best_checkpoint"])
    assert best_file.is_file()

    # Verifica métricas
    metrics = results["best_metrics"]
    assert "m1_residual_median_db" in metrics
    assert "m3_ident_mean_abs_g_db" in metrics
    assert "m4_consistency_mean_std_db" in metrics

    # Verifica exportação ONNX
    assert onnx_path.is_file()
    proto = onnx.load(str(onnx_path))
    assert proto.ir_version <= 8

    # Valida paridade numérica PyTorch vs ONNX Runtime (< 1e-4 dB)
    max_diff = validate_onnx_parity(model, onnx_path, tolerance_db=1e-4, test_batches=(1, 2, 5))
    assert max_diff < 1e-4

    # Verifica tract compatibility
    tract_ok = verify_tract_compatibility(onnx_path)
    assert tract_ok is True


def test_onnx_dynamic_batching(tmp_path: Path):
    """Valida inferência em lotes arbitrários com onnxruntime."""
    model = NeuralEqModel().eval()
    onnx_file = tmp_path / "test_dynamic.onnx"
    export_neural_eq_onnx(model, onnx_file)

    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    for b in (1, 3, 7, 13):
        inp = np.random.randn(b, NUM_ERB_BANDS, 3).astype(np.float32)
        out = sess.run(["gains_db"], {"feat": inp})[0]
        assert out.shape == (b, NUM_ERB_BANDS)
        assert np.all(out >= GAIN_MIN_DB - 1e-6)
        assert np.all(out <= GAIN_MAX_DB + 1e-6)
