"""Arquitetura de rede neural para predição de equalização espectral estática (Neural EQ - Subprojeto B).

Conforme especificações técnicas:
    - Entrada: feat [batch, 32, 3] ou [batch, 96] com estatísticas log-espectrais em bandas ERB.
    - Projeção inicial ultraleve: 96 -> 128 -> 128 -> 8 coeficientes (~30k parâmetros).
    - Expansão linear dos 8 coeficientes para 32 bandas ERB via base suave de cossenos (DCT-II).
    - Camada de saída com ativação limitada estritamente em [-6.0, +12.0] dB com zona neutra
      contínua para preservação de voz já equilibrada.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neural_eq.constants import (
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    NUM_ERB_BANDS,
    build_dct_basis,
)

NUM_COSINE_COEFFS: int = 8
FEAT_INPUT_DIM: int = 96  # 32 bandas x 3 features por banda


def build_cosine_basis(
    num_bands: int = NUM_ERB_BANDS,
    num_coeffs: int = NUM_COSINE_COEFFS,
    norm: str = "none",
) -> Tensor:
    """Gera a base suave de cossenos [num_bands, num_coeffs]."""
    if norm == "ortho":
        return build_dct_basis(num_bands=num_bands, num_modes=num_coeffs)

    n = torch.arange(num_bands, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(num_coeffs, dtype=torch.float32).unsqueeze(0)
    return torch.cos(torch.pi * k * (n + 0.5) / float(num_bands))


class BoundedNeutralActivation(nn.Module):
    """Ativação estritamente limitada em [-min_db, +max_db] dB com zona neutra contínua.

    Preserva voz já equilibrada mantendo 0 dB quando as alterações estão dentro da zona neutra,
    e transiciona suave e monotonicamente até os limites assintóticos com clamp rígido de segurança.
    """

    def __init__(
        self,
        min_db: float = 6.0,
        max_db: float = 12.0,
        neutral_zone: float = 0.5,
    ) -> None:
        super().__init__()
        if min_db <= 0.0 or max_db <= 0.0:
            raise ValueError(f"Limites de dB devem ser positivos, recebido min_db={min_db}, max_db={max_db}")
        if neutral_zone < 0.0:
            raise ValueError(f"neutral_zone deve ser >= 0, recebido {neutral_zone}")

        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.neutral_zone = float(neutral_zone)

    def forward(self, x: Tensor) -> Tensor:
        if self.neutral_zone > 0.0:
            # Zona neutra contínua via soft-shrinkage
            z = torch.sign(x) * F.relu(torch.abs(x) - self.neutral_zone)
        else:
            z = x

        # Saturação suave assimétrica com derivada C1 unitária na transição
        pos = self.max_db * torch.tanh(z / self.max_db)
        neg = self.min_db * torch.tanh(z / self.min_db)
        y = torch.where(z >= 0, pos, neg)

        return y.clamp(min=-self.min_db, max=self.max_db)


class NeuralEqNet(nn.Module):
    """Rede neural ultraleve (~30k parâmetros) para estimar 32 ganhos ERB.

    Arquitetura:
        - Projeção inicial 96 -> 128 (Linear + ReLU)
        - Camada intermediária 128 -> 128 (Linear + ReLU)
        - Projeção de coeficientes 128 -> 8 (Linear)
        - Expansão linear: 8 coeficientes -> 32 bandas ERB via base de cossenos suave
        - Ativação de saída: limitada estritamente em [-6.0, +12.0] dB com zona neutra contínua.
    """

    def __init__(
        self,
        in_features: int = FEAT_INPUT_DIM,
        hidden_dim: int = 128,
        num_coeffs: int = NUM_COSINE_COEFFS,
        num_bands: int = NUM_ERB_BANDS,
        min_db: float = 6.0,
        max_db: float = 12.0,
        neutral_zone: float = 0.5,
        init_zeros: bool = True,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_bands = num_bands
        self.num_coeffs = num_coeffs

        # Camadas densas: 96->128, 128->128, 128->8 (total 29.960 parâmetros)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_coeffs)

        # Base suave de cossenos registrada como buffer
        basis = build_cosine_basis(num_bands=num_bands, num_coeffs=num_coeffs, norm=norm)
        self.register_buffer("cosine_basis", basis)

        self.activation = BoundedNeutralActivation(
            min_db=min_db,
            max_db=max_db,
            neutral_zone=neutral_zone,
        )

        if init_zeros:
            nn.init.zeros_(self.fc_out.weight)
            nn.init.zeros_(self.fc_out.bias)

    def extract_coeffs(self, feat: Tensor) -> Tensor:
        """Extrai os 8 coeficientes da base de cossenos."""
        x = self._prepare_features(feat)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_out(h2)

    def _prepare_features(self, feat: Tensor) -> Tensor:
        if feat.ndim == 3 and feat.shape[-2:] == (self.num_bands, self.in_features // self.num_bands):
            return feat.reshape(feat.shape[0], -1)
        if feat.ndim >= 2 and feat.shape[-1] == self.in_features:
            return feat
        if feat.shape[-2:] == (self.num_bands, 3):
            orig_lead = feat.shape[:-2]
            return feat.reshape(*orig_lead, -1)
        raise ValueError(
            f"Formato de entrada incompatível com in_features={self.in_features}. "
            f"Recebido formato {tuple(feat.shape)}"
        )

    def forward(self, feat: Tensor, *, return_coeffs: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Calcula os 32 ganhos ERB em dB estritamente limitados em [-min_db, +max_db]."""
        coeffs = self.extract_coeffs(feat)

        # Expansão linear dos coeficientes para as 32 bandas ERB
        linear_gains = torch.matmul(coeffs, self.cosine_basis.t())

        # Ativação com zona neutra contínua
        gains = self.activation(linear_gains)

        if return_coeffs:
            return gains, coeffs
        return gains


@dataclass
class NeuralEqConfig:
    num_bands: int = NUM_ERB_BANDS
    num_features_per_band: int = 3
    hidden_dim: int = 128
    num_basis_modes: int = 8
    gain_min_db: float = GAIN_MIN_DB
    gain_max_db: float = GAIN_MAX_DB


class NeuralEqModel(nn.Module):
    """Modelo Neural EQ com cabeças de base DCT e portão de confiança."""

    def __init__(self, config: NeuralEqConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuralEqConfig()
        in_dim = self.config.num_bands * self.config.num_features_per_band

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
        )

        self.head_basis = nn.Linear(self.config.hidden_dim, self.config.num_basis_modes)
        self.head_confidence = nn.Linear(self.config.hidden_dim, 1)

        basis = build_dct_basis(self.config.num_bands, self.config.num_basis_modes)
        self.register_buffer("basis", basis)

        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            self.head_basis.weight.normal_(mean=0.0, std=0.01)
            self.head_basis.bias.zero_()
            self.head_confidence.weight.normal_(mean=0.0, std=0.01)
            self.head_confidence.bias.fill_(1.0)

    def forward(self, feat: Tensor) -> Tensor:
        batch_size = feat.shape[0]
        x = feat.reshape(batch_size, -1)
        h = self.net(x)

        weights = self.head_basis(h)
        raw_gains = torch.matmul(weights, self.basis.transpose(0, 1))

        conf = torch.sigmoid(self.head_confidence(h))
        scaled_gains = raw_gains * conf

        return torch.clamp(
            scaled_gains,
            min=self.config.gain_min_db,
            max=self.config.gain_max_db,
        )
