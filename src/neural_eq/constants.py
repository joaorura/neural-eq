"""Constantes e matrizes de interpolação espectral para o Neural EQ.

Segue as especificações de crates/model/src/spectral_eq.rs e docs/.../08-eq-neural.md:
- 32 bandas ERB (modelo 48 kHz / N=960).
- 481 bins FFT lineares (0 Hz a 24 kHz, 50 Hz por bin).
- Ganhos por banda no intervalo [-6.0, +12.0] dB.
- Matriz constante M (481 x 32) para interpolação linear em dB entre centros de banda.
"""

from __future__ import annotations

import numpy as np
import torch

NUM_ERB_BANDS: int = 32
TOTAL_FFT_BINS: int = 481
GAIN_MIN_DB: float = -6.0
GAIN_MAX_DB: float = 12.0

BAND_WIDTHS: list[int] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28,
    31, 37, 42, 50, 56, 67,
]


def compute_band_centers(widths: list[int] | None = None) -> np.ndarray:
    """Calcula a posição central fracionária de cada banda ERB em índices de bin FFT."""
    widths = BAND_WIDTHS if widths is None else widths
    centers = np.zeros(len(widths), dtype=np.float32)
    start = 0.0
    for k, w in enumerate(widths):
        centers[k] = start + (float(w) - 1.0) / 2.0
        start += float(w)
    return centers


def build_interpolation_matrix(widths: list[int] | None = None) -> torch.Tensor:
    """Constrói matriz M (481, 32) tal que r = g @ M.T interpola linearmente em dB.

    Reproduz bit a bit a lógica de `bin_factors` em crates/model/src/spectral_eq.rs:
    - Bins <= centro da banda 0 mantêm ganho da banda 0.
    - Bins >= centro da banda 31 mantêm ganho da banda 31.
    - Entre centros k e k+1, interpolação linear ponderada.
    """
    widths = BAND_WIDTHS if widths is None else widths
    total = sum(widths)
    centers = compute_band_centers(widths)
    num_bands = len(widths)

    m = np.zeros((total, num_bands), dtype=np.float32)
    for bin_idx in range(total):
        pos = float(bin_idx)
        if pos <= centers[0]:
            m[bin_idx, 0] = 1.0
        elif pos >= centers[-1]:
            m[bin_idx, -1] = 1.0
        else:
            k = 0
            while k + 1 < num_bands and centers[k + 1] <= pos:
                k += 1
            span = centers[k + 1] - centers[k]
            t = (pos - centers[k]) / span
            m[bin_idx, k] = 1.0 - t
            m[bin_idx, k + 1] = t

    return torch.from_numpy(m)


def build_dct_basis(num_bands: int = 32, num_modes: int = 8) -> torch.Tensor:
    """Base suave de cossenos (DCT-II ortonormal) de dimensão (num_bands, num_modes).

    Projeta K coeficientes suaves sobre as 32 bandas ERB garantindo ausência de descontinuidades.
    """
    basis = np.zeros((num_bands, num_modes), dtype=np.float32)
    for k in range(num_modes):
        scale = np.sqrt(1.0 / num_bands) if k == 0 else np.sqrt(2.0 / num_bands)
        for b in range(num_bands):
            basis[b, k] = scale * np.cos(np.pi * (float(b) + 0.5) * float(k) / float(num_bands))
    return torch.from_numpy(basis)
