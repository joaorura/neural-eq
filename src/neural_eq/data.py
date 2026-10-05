"""Geração sintética de distorções de canal e dataset para o Neural EQ.

Conforme spec 08-eq-neural.md §3.3:
- Amostragem de curvas d(f) em 481 bins (0-24 kHz) cobrindo:
  1. Identidade (d = 0, ~25% dos casos).
  2. EQ paramétrico (low/high shelf, peaking filters via biquad digital/analógico em dB).
  3. Colorimetrias típicas de microfones (efeito de proximidade, presença, ressonância em 400 Hz, roll-off).
  4. Curvas suaves aleatórias em espaço ERB.
  5. Família OOD (ressonâncias e degraus agudos) mantida estritamente para avaliação.
- Extração/simulação de features [32, 3] (média, P10, P90 da energia log em dB) após normalização de nível.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from neural_eq.constants import (
    BAND_WIDTHS,
    NUM_ERB_BANDS,
    TOTAL_FFT_BINS,
    build_interpolation_matrix,
)

# Perfil espectral médio de longo prazo (LTAS) aproximado de fala limpa de estúdio em 32 bandas ERB
STUDIO_SPEECH_LTAS = np.array([
    -18.0, -14.0, -10.0, -8.0, -7.0, -6.5, -6.0, -6.0,
    -6.5, -7.0, -7.5, -8.0, -9.0, -10.0, -11.5, -13.0,
    -14.5, -16.0, -18.0, -20.0, -22.0, -24.0, -26.5, -29.0,
    -31.5, -34.0, -37.0, -40.0, -43.0, -46.0, -49.0, -52.0,
], dtype=np.float32)


def generate_parametric_curve(
    num_bins: int = TOTAL_FFT_BINS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Gera curva de EQ paramétrico em dB sobre os 481 bins (50 Hz por bin, 0-24 kHz)."""
    rng = np.random.default_rng() if rng is None else rng
    freqs = np.linspace(0.0, 24_000.0, num_bins, dtype=np.float32)
    curve = np.zeros(num_bins, dtype=np.float32)

    # Low-shelf (100 a 400 Hz)
    if rng.random() > 0.3:
        fc = rng.uniform(80.0, 400.0)
        gain = rng.uniform(-10.0, 10.0)
        # Aproximação suave da resposta de shelf em dB
        curve += gain / (1.0 + (np.maximum(freqs, 10.0) / fc) ** 2)

    # High-shelf (4 kHz a 12 kHz)
    if rng.random() > 0.3:
        fc = rng.uniform(4000.0, 12000.0)
        gain = rng.uniform(-10.0, 10.0)
        curve += gain * ((freqs / fc) ** 2) / (1.0 + (freqs / fc) ** 2)

    # 1 a 3 filtros de pico / bell
    num_peaks = int(rng.integers(1, 4))
    for _ in range(num_peaks):
        f0 = rng.uniform(200.0, 8000.0)
        bw = rng.uniform(0.5, 2.0) * f0 * 0.3
        gain = rng.uniform(-8.0, 8.0)
        q_curve = np.exp(-0.5 * ((freqs - f0) / (bw + 1e-5)) ** 2)
        curve += gain * q_curve

    return curve


def generate_mic_coloration_curve(
    num_bins: int = TOTAL_FFT_BINS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Gera coloração característica de microfones comuns (proximidade, presença, corte)."""
    rng = np.random.default_rng() if rng is None else rng
    freqs = np.linspace(0.0, 24_000.0, num_bins, dtype=np.float32)
    curve = np.zeros(num_bins, dtype=np.float32)

    # Efeito de proximidade (grave inflado)
    if rng.random() > 0.4:
        boost = rng.uniform(2.0, 8.0)
        f_prox = rng.uniform(120.0, 250.0)
        curve += boost / (1.0 + (np.maximum(freqs, 20.0) / f_prox) ** 2)

    # Pico de presença de condensador/dinâmico vocal (2 a 6 kHz)
    if rng.random() > 0.3:
        presence = rng.uniform(2.0, 6.0)
        f_pres = rng.uniform(2500.0, 5000.0)
        curve += presence * np.exp(-0.5 * ((freqs - f_pres) / 1500.0) ** 2)

    # Som abafado / "boxy" (300 a 600 Hz)
    if rng.random() > 0.4:
        boxy = rng.uniform(-4.0, 5.0)
        curve += boxy * np.exp(-0.5 * ((freqs - 450.0) / 200.0) ** 2)

    # Roll-off de altas frequências (> 8-12 kHz)
    if rng.random() > 0.4:
        rolloff_f = rng.uniform(8000.0, 16000.0)
        slope = rng.uniform(-12.0, -3.0)
        high_mask = freqs > rolloff_f
        curve[high_mask] += slope * np.log2(freqs[high_mask] / rolloff_f)

    # Roll-off de sub-graves (< 80 Hz)
    hpf_f = rng.uniform(40.0, 100.0)
    sub_mask = freqs < hpf_f
    curve[sub_mask] -= rng.uniform(4.0, 12.0) * (1.0 - freqs[sub_mask] / hpf_f)

    return curve


def generate_smooth_erb_curve(
    num_bins: int = TOTAL_FFT_BINS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Gera distorção espectral suave aleatória por combinação de cossenos ERB."""
    rng = np.random.default_rng() if rng is None else rng
    m = build_interpolation_matrix().numpy()
    num_modes = int(rng.integers(2, 6))
    coeffs = rng.normal(0.0, 2.5, size=num_modes).astype(np.float32)
    band_gains = np.zeros(NUM_ERB_BANDS, dtype=np.float32)
    for k, c in enumerate(coeffs):
        band_gains += c * np.cos(np.pi * (np.arange(NUM_ERB_BANDS) + 0.5) * (k + 1) / 32.0)
    curve = m @ band_gains
    return curve.astype(np.float32)


def generate_ood_curve(
    num_bins: int = TOTAL_FFT_BINS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Família Out-of-Distribution (OOD): distribuição deslocada (4 a 6 picos, inclinações acentuadas).

    Análogo ao Validation Set 2 de Nercessian (DAFx 2020): curvas paramétricas fora da distribuição
    de treino para testar generalização sem degradação vs bypass.
    """
    rng = np.random.default_rng() if rng is None else rng
    freqs = np.linspace(0.0, 24_000.0, num_bins, dtype=np.float32)
    curve = np.zeros(num_bins, dtype=np.float32)

    # Low-shelf acentuado e deslocado
    fc_low = rng.uniform(60.0, 500.0)
    gain_low = rng.uniform(-7.0, 5.0)
    curve += gain_low / (1.0 + (np.maximum(freqs, 10.0) / fc_low) ** 2.5)

    # High-shelf acentuado
    fc_high = rng.uniform(3000.0, 10000.0)
    gain_high = rng.uniform(-7.0, 5.0)
    curve += gain_high * ((freqs / fc_high) ** 2.5) / (1.0 + (freqs / fc_high) ** 2.5)

    # 1 a 3 picos ressonantes com frequências e larguras fora da distribuição de treino
    num_peaks = int(rng.integers(1, 4))
    for _ in range(num_peaks):
        f0 = rng.uniform(80.0, 14000.0)
        bw = rng.uniform(0.15, 0.6) * f0 * 0.2
        gain = rng.uniform(-5.0, 5.0)
        q_curve = np.exp(-0.5 * ((freqs - f0) / (bw + 1e-5)) ** 2)
        curve += gain * q_curve

    return curve


def channel_bins_to_erb_features(
    distortion_bins: np.ndarray,
    rng: np.random.Generator | None = None,
    speaker_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Converte distorção nos 481 bins para features [32, 3] (média, P10, P90) simuladas."""
    rng = np.random.default_rng() if rng is None else rng
    widths = BAND_WIDTHS

    # Extrai o ganho médio por banda ERB
    band_distortions = np.zeros(NUM_ERB_BANDS, dtype=np.float32)
    start = 0
    for b, w in enumerate(widths):
        band_distortions[b] = float(np.mean(distortion_bins[start : start + w]))
        start += w

    # Variação do locutor em torno do LTAS médio de estúdio
    if speaker_shift is None:
        spk_variation = rng.normal(0.0, 1.8, size=NUM_ERB_BANDS).astype(np.float32)
    else:
        spk_variation = speaker_shift

    clean_band_energy = STUDIO_SPEECH_LTAS + spk_variation
    distorted_band_energy = clean_band_energy + band_distortions

    # Normalização de nível global (centraliza média das bandas vocais em 0)
    norm_offset = float(np.mean(distorted_band_energy[2:24]))
    distorted_band_energy -= norm_offset

    # Simulação da dinâmica temporal de quadros com fala (P10 e P90 em torno da média)
    spread = rng.uniform(4.0, 8.0, size=NUM_ERB_BANDS).astype(np.float32)
    noise = rng.normal(0.0, 0.4, size=NUM_ERB_BANDS).astype(np.float32)

    mean_feat = distorted_band_energy + noise
    p10_feat = mean_feat - spread * 0.5 + rng.normal(0.0, 0.3, size=NUM_ERB_BANDS).astype(np.float32)
    p90_feat = mean_feat + spread * 0.5 + rng.normal(0.0, 0.3, size=NUM_ERB_BANDS).astype(np.float32)

    # feat shape: [32, 3]
    feat = np.stack([mean_feat, p10_feat, p90_feat], axis=-1)
    return feat.astype(np.float32)


class SyntheticEqDataset(Dataset):
    """Dataset sintético de pares (feat, distortion_bins, is_identity) para treino e validação."""

    def __init__(
        self,
        num_samples: int = 5000,
        identity_ratio: float = 0.25,
        is_ood: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.identity_ratio = identity_ratio
        self.is_ood = is_ood
        self.rng = np.random.default_rng(seed)

        self.samples: list[tuple[np.ndarray, np.ndarray, bool]] = []
        self._generate_all()

    def _generate_all(self) -> None:
        for _ in range(self.num_samples):
            if self.is_ood:
                curve = generate_ood_curve(rng=self.rng)
                is_ident = False
            elif self.rng.random() < self.identity_ratio:
                curve = np.zeros(TOTAL_FFT_BINS, dtype=np.float32)
                is_ident = True
            else:
                fam = self.rng.choice([1, 2, 3])
                if fam == 1:
                    curve = generate_parametric_curve(rng=self.rng)
                elif fam == 2:
                    curve = generate_mic_coloration_curve(rng=self.rng)
                else:
                    curve = generate_smooth_erb_curve(rng=self.rng)
                is_ident = False

            # Limita a distorção para que o alvo caiba prioritariamente em [-12, +6] (inversa em [-6, +12])
            curve = np.clip(curve, -14.0, 8.0)
            feat = channel_bins_to_erb_features(curve, rng=self.rng)
            self.samples.append((feat, curve.astype(np.float32), is_ident))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, curve, is_ident = self.samples[idx]
        return (
            torch.from_numpy(feat),
            torch.from_numpy(curve),
            torch.tensor(is_ident, dtype=torch.bool),
        )


def create_consistency_evaluation_cases(
    num_channels: int = 20,
    variations_per_channel: int = 5,
    seed: int = 1234,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Gera casos para teste M4 (consistência): mesmo canal, múltiplos clipes com variação de locutor."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(num_channels):
        curve = generate_mic_coloration_curve(rng=rng)
        var_feats = []
        for _ in range(variations_per_channel):
            feat = channel_bins_to_erb_features(curve, rng=rng)
            var_feats.append(torch.from_numpy(feat))
        batch_feats = torch.stack(var_feats, dim=0)  # [variations, 32, 3]
        cases.append((batch_feats, torch.from_numpy(curve)))
    return cases
