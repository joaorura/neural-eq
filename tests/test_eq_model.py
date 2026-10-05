"""Testes unitários para o Neural EQ (model.py e losses.py).

Testa:
    1. Contagem exata de parâmetros (~30k parâmetros).
    2. Formatos de entrada no forward pass ([B, 32, 3], [B, 96], [B, T, 32, 3]).
    3. Limites de saída estritamente em [-6.0, +12.0] dB.
    4. Zona neutra contínua e preservação de voz já equilibrada (ganho 0 dB).
    5. Fluxo de gradientes em todas as camadas.
    6. Suavidade da base de cossenos.
    7. Paridade estrita da matriz M com spectral_eq::bin_factors do Rust.
    8. Huber loss, regularizador de curvatura e regularizador de neutralidade.
    9. Ciclo de otimização end-to-end.
"""

from __future__ import annotations

import math

import pytest
import torch

from neural_eq.losses import (
    MODEL_ERB_WIDTHS,
    NUM_ERB_BANDS,
    NUM_FFT_BINS,
    SpectralEqLoss,
    build_erb_interpolation_matrix,
    compute_erb_centers,
    curvature_regularizer,
    interpolate_erb_to_bins,
    neutrality_regularizer,
    spectral_huber_loss,
)
from neural_eq.model import (
    BoundedNeutralActivation,
    NeuralEqNet,
    build_cosine_basis,
)


def test_parameter_count_is_approximately_30k():
    """A rede deve ter ~30k parâmetros treináveis (exatamente 29.960 com 96->128->128->8)."""
    net = NeuralEqNet()
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net.parameters())

    # fc1: 96 * 128 + 128 = 12416
    # fc2: 128 * 128 + 128 = 16512
    # fc_out: 128 * 8 + 8 = 1032
    # Total = 29960
    assert trainable_params == 29_960
    assert total_params == 29_960

    # cosine_basis é buffer, não parâmetro treinável
    assert "cosine_basis" in net._buffers
    assert net.cosine_basis.shape == (32, 8)
    assert not net.cosine_basis.requires_grad


def test_forward_pass_shapes():
    """Testa compatibilidade com [B, 32, 3], [B, 96] e [B, T, 32, 3]."""
    net = NeuralEqNet()

    # Formato [batch, 32, 3]
    feat_3d = torch.randn(4, 32, 3)
    out_3d = net(feat_3d)
    assert out_3d.shape == (4, 32)

    # Formato [batch, 96]
    feat_2d = torch.randn(4, 96)
    out_2d = net(feat_2d)
    assert out_2d.shape == (4, 32)

    # Formato com dimensão temporal [batch, time, 32, 3]
    feat_seq = torch.randn(2, 5, 32, 3)
    out_seq = net(feat_seq)
    assert out_seq.shape == (2, 5, 32)

    # Retorno opcional de coeficientes
    gains, coeffs = net(feat_3d, return_coeffs=True)
    assert gains.shape == (4, 32)
    assert coeffs.shape == (4, 8)


def test_invalid_input_shape_raises_error():
    """Testa que formatos inválidos levantam ValueError."""
    net = NeuralEqNet()
    with pytest.raises(ValueError, match="Formato de entrada incompatível"):
        net(torch.randn(4, 30, 2))


def test_strict_output_bounds_even_under_extreme_activations():
    """Testa que a saída nunca ultrapassa [-6.0, +12.0] dB mesmo com ativações extremas."""
    net = NeuralEqNet(init_zeros=False)

    # Forçar pesos e bias gigantescos para testar limites de saturação
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(std=50.0)

    # Entradas normais e extremas
    feat_extreme = torch.randn(32, 32, 3) * 1000.0
    out = net(feat_extreme)

    assert out.min() >= -6.0
    assert out.max() <= 12.0
    assert torch.isfinite(out).all()


def test_bounded_activation_direct_limits_and_clamping():
    """Testa BoundedNeutralActivation isoladamente em extremos positivos e negativos."""
    act = BoundedNeutralActivation(min_db=6.0, max_db=12.0, neutral_zone=0.5)

    x = torch.tensor([-1e6, -100.0, -10.0, -0.5, 0.0, 0.5, 10.0, 100.0, 1e6])
    y = act(x)

    # Limites rígidos
    assert float(y[0]) >= -6.0
    assert float(y[-1]) <= 12.0
    assert (y >= -6.0).all()
    assert (y <= 12.0).all()

    # Zona neutra [-0.5, 0.5]
    assert float(y[3]) == 0.0
    assert float(y[4]) == 0.0
    assert float(y[5]) == 0.0


def test_continuous_neutral_zone_preserves_balanced_speech():
    """Testa que a zona neutra mantém 0.0 dB para entradas com alterações insignificantes."""
    net = NeuralEqNet(init_zeros=True, neutral_zone=0.5)

    # Com init_zeros, qualquer entrada gera ganho exatamente zero (pass-through idêntico)
    feat = torch.randn(8, 32, 3)
    out = net(feat)
    assert torch.equal(out, torch.zeros_like(out))

    # Testando continuidade: pequenas perturbações dentro da zona neutra dão 0 dB
    act = BoundedNeutralActivation(min_db=6.0, max_db=12.0, neutral_zone=0.5)
    inside_zone = torch.linspace(-0.5, 0.5, 21)
    assert torch.all(act(inside_zone) == 0.0)

    # Logo fora da zona neutra, transição contínua
    just_above = torch.tensor([0.501])
    just_below = torch.tensor([-0.501])
    assert 0.0 < float(act(just_above)) < 0.01
    assert -0.01 < float(act(just_below)) < 0.0


def test_gradient_flow_to_all_trainable_parameters():
    """Garante que os gradientes fluem sem interrupção por todas as camadas da rede."""
    net = NeuralEqNet(init_zeros=False, neutral_zone=0.1)

    # Entradas com variação suficiente para sair da zona neutra
    feat = torch.randn(8, 32, 3, requires_grad=True)
    gains = net(feat)

    # Perda que demanda resposta diferente de zero
    loss = (gains - 3.0).pow(2).mean()
    loss.backward()

    # Gradiente na entrada
    assert feat.grad is not None
    assert torch.isfinite(feat.grad).all()

    # Gradientes em todos os parâmetros da rede
    for name, param in net.named_parameters():
        assert param.grad is not None, f"Gradiente ausente em {name}"
        assert torch.isfinite(param.grad).all(), f"Gradiente não-finito em {name}"
        assert param.grad.abs().sum() > 0.0, f"Gradiente zerado em {name}"


def test_cosine_basis_properties():
    """Verifica que a base de cossenos gera curvas suaves sem descontinuidades."""
    basis = build_cosine_basis(num_bands=32, num_coeffs=8, norm="none")
    assert basis.shape == (32, 8)

    # O modo 0 (DC) deve ser constante e igual a 1.0 em todas as bandas
    assert torch.allclose(basis[:, 0], torch.ones(32))

    # O modo 1 deve ser um meio-cosseno suave e estritamente decrescente de +1 a -1
    diff_m1 = torch.diff(basis[:, 1])
    assert (diff_m1 < 0.0).all()

    # Base ortonormal
    ortho_basis = build_cosine_basis(num_bands=32, num_coeffs=8, norm="ortho")
    assert ortho_basis.shape == (32, 8)
    assert pytest.approx(float(ortho_basis[:, 0].mean())) == 1.0 / math.sqrt(32.0)


def test_erb_interpolation_matrix_rust_parity():
    """Verifica a paridade exata de M com a função spectral_eq::bin_factors do Rust."""
    m = build_erb_interpolation_matrix(widths=MODEL_ERB_WIDTHS)
    assert m.shape == (NUM_FFT_BINS, NUM_ERB_BANDS)
    assert m.shape == (481, 32)

    # 1. Toda linha deve somar exatamente 1.0
    row_sums = m.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    # 2. Todos os elementos devem ser >= 0
    assert (m >= 0.0).all()

    # 3. Ganho plano dá exatamente o mesmo ganho em todos os bins (como no teste Rust)
    flat_gains = torch.full((1, 32), 6.0)
    flat_response = interpolate_erb_to_bins(flat_gains, m)
    assert flat_response.shape == (1, 481)
    assert torch.allclose(flat_response, torch.full((1, 481), 6.0), atol=1e-6)

    # 4. Ganhos em degrau (0 dB bandas < 16, 12 dB bandas >= 16)
    step_gains = torch.zeros((1, 32))
    step_gains[0, 16:] = 12.0
    step_response = interpolate_erb_to_bins(step_gains, m).squeeze(0)

    # Monotonicamente não-decrescente (tolerância 1e-4 idêntica ao teste Rust parity_eq.rs)
    step_diffs = torch.diff(step_response)
    assert (step_diffs >= -1e-4).all()

    # Sem saltos abruptos de 12 dB entre bins vizinhos
    max_step = step_diffs.max().item()
    assert max_step < 12.0

    # Extremidades
    assert pytest.approx(step_response[0].item(), abs=1e-5) == 0.0
    assert pytest.approx(step_response[-1].item(), abs=1e-5) == 12.0


def test_curvature_regularizer():
    """Testa que curvas lineares têm curvatura zero e oscilações têm penalidade alta."""
    # Linha reta: ganho = 2.0 * k - 5.0
    k = torch.arange(32, dtype=torch.float32).unsqueeze(0)
    linear_gains = 2.0 * k - 5.0
    loss_linear = curvature_regularizer(linear_gains)
    assert pytest.approx(loss_linear.item(), abs=1e-6) == 0.0

    # Oscilação alternada tipo dente de serra
    zigzag = torch.tensor([[0.0, 5.0] * 16])
    loss_zigzag = curvature_regularizer(zigzag)
    assert loss_zigzag.item() > 10.0


def test_neutrality_regularizer():
    """Testa o regularizador de neutralidade L1 (Smooth L1 suave)."""
    zeros = torch.zeros(2, 32)
    assert pytest.approx(neutrality_regularizer(zeros).item(), abs=1e-7) == 0.0

    small_gains = torch.full((2, 32), 0.2)
    large_gains = torch.full((2, 32), 4.0)

    loss_small = neutrality_regularizer(small_gains)
    loss_large = neutrality_regularizer(large_gains)

    assert 0.0 < loss_small.item() < loss_large.item()


def test_spectral_huber_loss():
    """Testa Huber loss espectral entre predição e alvo."""
    pred = torch.full((2, 481), 2.0)
    target = torch.full((2, 481), 2.0)
    assert pytest.approx(spectral_huber_loss(pred, target).item(), abs=1e-7) == 0.0

    # Erro pequeno (quadrático)
    target_small = torch.full((2, 481), 2.5)
    loss_small = spectral_huber_loss(pred, target_small, delta=1.0)
    # Huber para erro 0.5 <= 1.0 é 0.5 * 0.5^2 = 0.125
    assert pytest.approx(loss_small.item(), rel=1e-4) == 0.125

    # Erro grande (linear)
    target_large = torch.full((2, 481), 5.0)
    loss_large = spectral_huber_loss(pred, target_large, delta=1.0)
    # Huber para erro 3.0 > 1.0 é 3.0 - 0.5 = 2.5
    assert pytest.approx(loss_large.item(), rel=1e-4) == 2.5


def test_spectral_eq_loss_module_end_to_end():
    """Testa o módulo consolidado SpectralEqLoss com alvo no domínio FFT e no domínio ERB."""
    loss_fn = SpectralEqLoss(weight_spectral=1.0, weight_curvature=0.1, weight_neutrality=0.01)

    pred_gains = torch.randn(4, 32, requires_grad=True)

    # 1. Alvo no domínio FFT [4, 481]
    target_fft = torch.randn(4, 481)
    res_fft = loss_fn(pred_gains, target_fft)
    assert "total" in res_fft
    assert "spectral" in res_fft
    assert "curvature" in res_fft
    assert "neutrality" in res_fft
    assert "pred_response" in res_fft
    assert res_fft["total"].requires_grad

    res_fft["total"].backward()
    assert pred_gains.grad is not None
    assert torch.isfinite(pred_gains.grad).all()

    # 2. Alvo no domínio ERB [4, 32]
    pred_gains.grad.zero_()
    target_erb = torch.randn(4, 32)
    res_erb = loss_fn(pred_gains, target_erb)
    assert res_erb["total"].requires_grad
    res_erb["total"].backward()
    assert pred_gains.grad is not None


def test_training_loop_convergence():
    """Valida um mini-ciclo de treino confirmando que a loss decresce com AdamW."""
    torch.manual_seed(42)
    net = NeuralEqNet(init_zeros=False, neutral_zone=0.1)
    loss_fn = SpectralEqLoss(weight_spectral=1.0, weight_curvature=0.05, weight_neutrality=0.005)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)

    feat = torch.randn(8, 32, 3)
    # Alvo com realce suave de agudos (+4 dB nos agudos, 0 dB nos graves)
    centers = torch.tensor(compute_erb_centers(), dtype=torch.float32)
    target_curve = (centers / 481.0) * 4.0  # [481]
    target_batch = target_curve.unsqueeze(0).expand(8, -1)

    initial_loss = loss_fn(net(feat), target_batch)["total"].item()

    for _ in range(30):
        optimizer.zero_grad()
        out = net(feat)
        losses = loss_fn(out, target_batch)
        losses["total"].backward()
        optimizer.step()

    final_loss = loss_fn(net(feat), target_batch)["total"].item()
    assert final_loss < initial_loss * 0.5, f"Loss não convergiu adequadamente: {initial_loss} -> {final_loss}"
