"""Funções de perda para o treinamento do Neural EQ no domínio da resposta realizada.

Conforme spec 08-eq-neural.md §3.4:
- Resposta realizada: r = pred_gains @ M.T [batch, 481]
- Resíduo de canal: e = d + r [batch, 481]
- Perda Huber assimétrica: penaliza mais a sobrecorreção / piora em relação ao bypass
- Regularização de não-agressão em identidade (d = 0)
- Regularização de suavidade (segunda diferença) e média nula de ganhos
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neural_eq.constants import build_interpolation_matrix


class NeuralEqLoss(nn.Module):
    """Perda no domínio da resposta espectral para o Neural EQ."""

    def __init__(
        self,
        weight_residual: float = 1.0,
        weight_asymmetry: float = 0.5,
        weight_identity: float = 2.0,
        weight_smoothness: float = 0.1,
        weight_l1_gain: float = 0.01,
        weight_mean_zero: float = 0.05,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_residual = weight_residual
        self.weight_asymmetry = weight_asymmetry
        self.weight_identity = weight_identity
        self.weight_smoothness = weight_smoothness
        self.weight_l1_gain = weight_l1_gain
        self.weight_mean_zero = weight_mean_zero
        self.huber_delta = huber_delta

        # Matriz M (481, 32)
        m = build_interpolation_matrix()
        self.register_buffer("interp_matrix", m)

    def forward(
        self,
        pred_gains: Tensor,
        distortion_bins: Tensor,
        is_identity: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Calcula perdas do Neural EQ.

        Args:
            pred_gains: Tensor [batch, 32] com os ganhos preditos em dB.
            distortion_bins: Tensor [batch, 481] com a distorção do canal em dB.
            is_identity: Tensor [batch] booleano indicando clipes com d = 0.

        Returns:
            Dicionário com 'loss_total', 'loss_residual', 'loss_identity', etc.
        """
        # Resposta realizada r nos 481 bins FFT: [batch, 481]
        realized_r = torch.matmul(pred_gains, self.interp_matrix.transpose(0, 1))

        # Erro residual e = d + r (o alvo é r = -d, logo e = 0)
        residual = distortion_bins + realized_r

        # Perda Huber no resíduo
        abs_res = torch.abs(residual)
        huber = torch.where(
            abs_res <= self.huber_delta,
            0.5 * (abs_res**2),
            self.huber_delta * (abs_res - 0.5 * self.huber_delta),
        )
        loss_res = huber.mean()

        # Penalidade de assimetria: quando |d + r| > |d| (o EQ aumentou o desvio em vez de reduzir)
        bypass_abs = torch.abs(distortion_bins)
        worse_mask = (abs_res > bypass_abs).float()
        loss_asym = (worse_mask * (abs_res - bypass_abs)).mean()

        # Perda de não-agressão para curvas identidade (d = 0)
        if is_identity is not None and is_identity.any():
            ident_mask = is_identity.float().unsqueeze(-1)  # [batch, 1]
            ident_gains = pred_gains * ident_mask
            loss_ident = (torch.abs(ident_gains).sum()) / (ident_mask.sum() * pred_gains.shape[-1] + 1e-8)
        else:
            loss_ident = torch.tensor(0.0, device=pred_gains.device)

        # Suavidade espectral (segunda diferença dos ganhos entre bandas vizinhas)
        if pred_gains.shape[-1] >= 3:
            diff2 = pred_gains[:, 2:] - 2.0 * pred_gains[:, 1:-1] + pred_gains[:, :-2]
            loss_smooth = torch.abs(diff2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=pred_gains.device)

        # Regularizador L1 em g (viés para neutro) e média próxima de 0 (nível tratado pelo AGC)
        loss_l1 = torch.abs(pred_gains).mean()
        loss_mean = torch.abs(pred_gains.mean(dim=-1)).mean()

        loss_total = (
            self.weight_residual * loss_res
            + self.weight_asymmetry * loss_asym
            + self.weight_identity * loss_ident
            + self.weight_smoothness * loss_smooth
            + self.weight_l1_gain * loss_l1
            + self.weight_mean_zero * loss_mean
        )

        return {
            "loss_total": loss_total,
            "loss_residual": loss_res,
            "loss_asym": loss_asym,
            "loss_identity": loss_ident,
            "loss_smooth": loss_smooth,
            "loss_l1": loss_l1,
            "loss_mean": loss_mean,
            "mae_residual_db": abs_res.mean(),
        }
