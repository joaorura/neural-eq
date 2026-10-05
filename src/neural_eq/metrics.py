"""Métricas de avaliação e validação contínua para o Neural EQ.

Implementa as métricas M1 a M4 conforme especificado em docs/.../08-eq-neural.md §3.5:
- M1: Erro residual da resposta |d + r| (dB): mediana <= 1.5 dB, P90 <= 3.0 dB.
- M2: Robustez OOD: piora relativa <= 50% vs in-distribution e sem degradação vs bypass.
- M3: Princípio 'Não machuque' (curvas d = 0): mean(|g|) <= 0.75 dB, P95(|g|) <= 2.0 dB.
- M4: Consistência de cadastro: desvio padrão dos ganhos <= 1.0 dB para mesmo canal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_eq.constants import build_interpolation_matrix
from neural_eq.model import NeuralEqModel


def evaluate_eq_metrics(
    model: NeuralEqModel,
    val_loader: DataLoader,
    ood_loader: DataLoader | None = None,
    ident_loader: DataLoader | None = None,
    consistency_cases: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Avalia o modelo Neural EQ nas métricas M1 a M4 da spec.

    Returns:
        Dicionário com métricas calculadas e flags de aprovação.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)
    interp_m = build_interpolation_matrix().to(device)

    all_res_abs: list[float] = []
    all_bypass_abs: list[float] = []
    ident_gains_abs: list[float] = []

    with torch.no_grad():
        # Avaliação no conjunto In-Distribution
        for feat, curve, is_ident in val_loader:
            feat = feat.to(device)
            curve = curve.to(device)
            is_ident = is_ident.to(device)

            pred_gains = model(feat)  # [B, 32]
            realized_r = torch.matmul(pred_gains, interp_m.transpose(0, 1))  # [B, 481]
            res = curve + realized_r  # [B, 481]

            res_abs = torch.abs(res).cpu().numpy().flatten()
            byp_abs = torch.abs(curve).cpu().numpy().flatten()

            all_res_abs.extend(res_abs.tolist())
            all_bypass_abs.extend(byp_abs.tolist())

            if is_ident.any():
                g_ident = torch.abs(pred_gains[is_ident]).cpu().numpy().flatten()
                ident_gains_abs.extend(g_ident.tolist())

    res_arr = np.array(all_res_abs, dtype=np.float32)
    byp_arr = np.array(all_bypass_abs, dtype=np.float32)

    # M1: Erro residual in-distribution
    m1_median = float(np.median(res_arr))
    m1_p90 = float(np.percentile(res_arr, 90))
    m1_mean = float(np.mean(res_arr))
    m1_bypass_mean = float(np.mean(byp_arr))
    m1_improvement = float((m1_bypass_mean - m1_mean) / (m1_bypass_mean + 1e-8) * 100.0)
    m1_pass = (m1_median <= 1.5) and (m1_p90 <= 3.0) and (m1_mean < m1_bypass_mean)

    # M2: Out-Of-Distribution (OOD)
    m2_mean = 0.0
    m2_median = 0.0
    m2_p90 = 0.0
    m2_bypass_mean = 0.0
    m2_rel_worsening = 0.0
    m2_pass = True

    if ood_loader is not None:
        ood_res_abs: list[float] = []
        ood_byp_abs: list[float] = []
        with torch.no_grad():
            for feat, curve, _ in ood_loader:
                feat = feat.to(device)
                curve = curve.to(device)
                pred_gains = model(feat)
                realized_r = torch.matmul(pred_gains, interp_m.transpose(0, 1))
                res = curve + realized_r
                ood_res_abs.extend(torch.abs(res).cpu().numpy().flatten().tolist())
                ood_byp_abs.extend(torch.abs(curve).cpu().numpy().flatten().tolist())

        ood_arr = np.array(ood_res_abs, dtype=np.float32)
        ood_byp = np.array(ood_byp_abs, dtype=np.float32)
        m2_mean = float(np.mean(ood_arr))
        m2_median = float(np.median(ood_arr))
        m2_p90 = float(np.percentile(ood_arr, 90))
        m2_bypass_mean = float(np.mean(ood_byp))
        m2_rel_worsening = float((m2_mean - m1_mean) / (m1_mean + 1e-8) * 100.0)
        # Spec §3.5: piora relativa <= 50% vs in-dist, e nunca pior que bypass
        m2_pass = (m2_rel_worsening <= 50.0) and (m2_mean <= m2_bypass_mean)

    # M3: 'Não machuque' (identidade d = 0)
    if ident_loader is not None and not ident_gains_abs:
        with torch.no_grad():
            for feat, _, _ in ident_loader:
                feat = feat.to(device)
                pred_gains = model(feat)
                ident_gains_abs.extend(torch.abs(pred_gains).cpu().numpy().flatten().tolist())

    if ident_gains_abs:
        ident_arr = np.array(ident_gains_abs, dtype=np.float32)
        m3_mean_abs_g = float(np.mean(ident_arr))
        m3_p95_abs_g = float(np.percentile(ident_arr, 95))
        m3_max_abs_g = float(np.max(ident_arr))
    else:
        m3_mean_abs_g = 0.0
        m3_p95_abs_g = 0.0
        m3_max_abs_g = 0.0
    # Spec §3.5: mean(|g|) <= 0.75 dB; P95(|g|) <= 2.0 dB
    m3_pass = (m3_mean_abs_g <= 0.75) and (m3_p95_abs_g <= 2.0)

    # M4: Consistência de cadastro (desvio padrão dos ganhos para mesmo canal <= 1.0 dB)
    m4_mean_std = 0.0
    m4_max_std = 0.0
    m4_pass = True

    if consistency_cases:
        stds = []
        with torch.no_grad():
            for batch_feats, _ in consistency_cases:
                batch_feats = batch_feats.to(device)  # [variations, 32, 3]
                pred_gains = model(batch_feats)  # [variations, 32]
                channel_std = torch.std(pred_gains, dim=0).mean().item()
                stds.append(channel_std)
        if stds:
            m4_mean_std = float(np.mean(stds))
            m4_max_std = float(np.max(stds))
            m4_pass = m4_mean_std <= 1.0

    return {
        "m1_residual_median_db": m1_median,
        "m1_residual_p90_db": m1_p90,
        "m1_residual_mean_db": m1_mean,
        "m1_bypass_mean_db": m1_bypass_mean,
        "m1_improvement_pct": m1_improvement,
        "m1_pass": m1_pass,
        "m2_ood_mean_db": m2_mean,
        "m2_ood_median_db": m2_median,
        "m2_ood_p90_db": m2_p90,
        "m2_ood_bypass_mean_db": m2_bypass_mean,
        "m2_rel_worsening_pct": m2_rel_worsening,
        "m2_pass": m2_pass,
        "m3_ident_mean_abs_g_db": m3_mean_abs_g,
        "m3_ident_p95_abs_g_db": m3_p95_abs_g,
        "m3_ident_max_abs_g_db": m3_max_abs_g,
        "m3_pass": m3_pass,
        "m4_consistency_mean_std_db": m4_mean_std,
        "m4_consistency_max_std_db": m4_max_std,
        "m4_pass": m4_pass,
        "all_metrics_pass": m1_pass and m2_pass and m3_pass and m4_pass,
    }
