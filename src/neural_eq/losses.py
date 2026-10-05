"""Perdas espectrais e regularizadores para o Neural EQ (Subprojeto B).

Implementa:
    1. Interpolação linear dos ganhos ERB para os 481 bins FFT (r = M * g), com paridade
       estrita com a função Rust `spectral_eq::bin_factors`.
    2. Huber Loss espectral entre a resposta estimada e a curva inversa alvo.
    3. Regularizador de curvatura (segunda diferença finita entre bandas adjacentes).
    4. Regularizador de neutralidade L1 (Smooth L1 puxando para 0 dB).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MODEL_ERB_WIDTHS: tuple[int, ...] = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37,
    42, 50, 56, 67,
)
NUM_ERB_BANDS: int = len(MODEL_ERB_WIDTHS)  # 32 bandas ERB
NUM_FFT_BINS: int = sum(MODEL_ERB_WIDTHS)   # 481 bins FFT (sr=48000, n_fft=960)


def compute_erb_centers(widths: Sequence[int] = MODEL_ERB_WIDTHS) -> list[float]:
    """Calcula os centros contínuos de frequência (em índices de bin) para cada banda ERB."""
    centers: list[float] = []
    start = 0
    for w in widths:
        centers.append(float(start) + (float(w) - 1.0) / 2.0)
        start += w
    return centers


def build_erb_interpolation_matrix(
    widths: Sequence[int] = MODEL_ERB_WIDTHS,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Constrói a matriz de interpolação M de dimensão [NUM_FFT_BINS, NUM_ERB_BANDS] (481 x 32).

    A multiplicação `r = gains @ M.t()` interpola linearmente os ganhos em dB entre os
    centros das bandas ERB, sendo 100% equivalente em aritmética e bordas à função Rust
    `spectral_eq::bin_factors`. Cada linha de M tem soma exatamente 1.0.
    """
    total = sum(widths)
    num_bands = len(widths)
    centers = compute_erb_centers(widths)

    m = torch.zeros((total, num_bands), dtype=dtype, device=device)
    band = 0
    for bin_idx in range(total):
        pos = float(bin_idx)
        while band + 1 < num_bands and centers[band + 1] <= pos:
            band += 1
        if pos <= centers[0] or band + 1 == num_bands:
            m[bin_idx, band] = 1.0
        else:
            t = (pos - centers[band]) / (centers[band + 1] - centers[band])
            m[bin_idx, band] = 1.0 - t
            m[bin_idx, band + 1] = t

    return m


def interpolate_erb_to_bins(gains: Tensor, matrix: Tensor) -> Tensor:
    """Interpola os ganhos ERB [..., 32] em dB para a resposta espectral [..., 481] em dB."""
    return torch.matmul(gains, matrix.t())


def spectral_huber_loss(
    pred_response: Tensor,
    target_response: Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Huber Loss entre a resposta estimada pred_response e a curva alvo target_response (ambas em dB)."""
    return F.huber_loss(pred_response, target_response, delta=delta, reduction=reduction)


def curvature_regularizer(
    gains: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Regularizador de curvatura (segunda diferença finita entre bandas ERB adjacentes).

    Delta^2 g[k] = g[k+2] - 2*g[k+1] + g[k]
    Penaliza variações abruptas e descontinuidades entre bandas vizinhas.
    Para qualquer reta (a*k + b), o valor é estritamente zero.
    """
    diff2 = torch.diff(gains, n=2, dim=-1)
    sq = diff2.pow(2)
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    raise ValueError(f"Redução desconhecida '{reduction}'. Use 'mean', 'sum' ou 'none'.")


def neutrality_regularizer(
    gains: Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Regularizador de neutralidade L1 (Smooth L1 suave puxando os ganhos para 0 dB).

    - Para |g| <= beta: 0.5 * g^2 / beta (suave e quadrático, sem cúspide não-diferenciável)
    - Para |g| > beta: |g| - 0.5 * beta (penalidade L1 linear)
    """
    zeros = torch.zeros_like(gains)
    return F.smooth_l1_loss(gains, zeros, beta=beta, reduction=reduction)


class SpectralEqLoss(nn.Module):
    """Módulo de perda consolidada para o treinamento do Neural EQ.

    Combina:
        1. Huber loss espectral sobre os 481 bins da FFT.
        2. Regularizador de curvatura (segunda diferença das bandas ERB).
        3. Regularizador de neutralidade L1 (Smooth L1 para 0 dB).
    """

    def __init__(
        self,
        widths: Sequence[int] = MODEL_ERB_WIDTHS,
        huber_delta: float = 1.0,
        smooth_l1_beta: float = 1.0,
        weight_spectral: float = 1.0,
        weight_curvature: float = 0.1,
        weight_neutrality: float = 0.01,
    ) -> None:
        super().__init__()
        self.huber_delta = float(huber_delta)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.weight_spectral = float(weight_spectral)
        self.weight_curvature = float(weight_curvature)
        self.weight_neutrality = float(weight_neutrality)

        matrix = build_erb_interpolation_matrix(widths=widths)
        self.register_buffer("erb_matrix", matrix)

    def forward(
        self,
        pred_gains: Tensor,
        target: Tensor,
    ) -> dict[str, Tensor]:
        """Calcula a perda total e cada um de seus componentes.

        Args:
            pred_gains: Ganhos ERB preditos [..., 32] em dB.
            target: Curva inversa alvo [..., 481] (no domínio FFT) ou [..., 32] (no domínio ERB).

        Returns:
            dict com 'total', 'spectral', 'curvature', 'neutrality' e 'pred_response'.
        """
        pred_response = interpolate_erb_to_bins(pred_gains, self.erb_matrix)

        if target.shape[-1] == self.erb_matrix.shape[0]:
            target_response = target
        elif target.shape[-1] == self.erb_matrix.shape[1]:
            target_response = interpolate_erb_to_bins(target, self.erb_matrix)
        else:
            raise ValueError(
                f"Dimensão de target incompatível: esperado {self.erb_matrix.shape[0]} (bins) "
                f"ou {self.erb_matrix.shape[1]} (ERB), recebido formato {tuple(target.shape)}"
            )

        loss_spectral = spectral_huber_loss(
            pred_response,
            target_response,
            delta=self.huber_delta,
        )
        loss_curv = curvature_regularizer(pred_gains)
        loss_neut = neutrality_regularizer(pred_gains, beta=self.smooth_l1_beta)

        total_loss = (
            self.weight_spectral * loss_spectral
            + self.weight_curvature * loss_curv
            + self.weight_neutrality * loss_neut
        )

        return {
            "total": total_loss,
            "spectral": loss_spectral,
            "curvature": loss_curv,
            "neutrality": loss_neut,
            "pred_response": pred_response,
        }
