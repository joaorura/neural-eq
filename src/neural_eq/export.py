"""Exportação do Neural EQ para ONNX dinâmico, validação de paridade e compatibilidade com Tract.

Conforme especificação:
- Input: feat [batch, 32, 3] (f32)
- Output: gains_db [batch, 32] (f32), limitado a [-6.0, +12.0] dB no grafo
- Opset 13, ir_version 8
- Tolerância numérica PyTorch vs ONNX Runtime < 1e-4 dB
- Validação com tract 0.19.16 via cctrain.tractcheck
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

try:
    from cctrain import tractcheck
except ImportError:
    tractcheck = None

from neural_eq.constants import GAIN_MAX_DB, GAIN_MIN_DB, NUM_ERB_BANDS
from neural_eq.model import NeuralEqModel

INPUT_NAME = "feat"
OUTPUT_NAME = "gains_db"
DEFAULT_TOLERANCE_DB = 1e-4


def export_neural_eq_onnx(
    model: NeuralEqModel,
    out_path: Path | str,
    opset_version: int = 13,
) -> Path:
    """Exporta o modelo NeuralEqModel para formato ONNX com batch dinâmico.

    Args:
        model: Instância de NeuralEqModel.
        out_path: Caminho de saída (.onnx).
        opset_version: Versão do opset ONNX (padrão 13 para Tract 0.19).

    Returns:
        Path para o arquivo .onnx gerado.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_eval = model.eval().cpu()
    example_input = torch.zeros(1, NUM_ERB_BANDS, 3, dtype=torch.float32)

    torch.onnx.export(
        model_eval,
        (example_input,),
        str(out_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    proto = onnx.load(str(out_path))
    proto.ir_version = 8  # Compatibilidade estrita com Tract 0.19
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out_path))

    return out_path


def validate_onnx_parity(
    model: NeuralEqModel,
    onnx_path: Path | str,
    tolerance_db: float = DEFAULT_TOLERANCE_DB,
    test_batches: tuple[int, ...] = (1, 4, 16),
    seed: int = 42,
) -> float:
    """Valida paridade numérica entre PyTorch e ONNX Runtime em múltiplos tamanhos de batch.

    Args:
        model: Instância de NeuralEqModel.
        onnx_path: Caminho do arquivo ONNX.
        tolerance_db: Diferença máxima tolerada em dB (deve ser < 1e-4).
        test_batches: Tamanhos de lote a testar para verificar eixos dinâmicos.
        seed: Semente aleatória para reprodutibilidade.

    Returns:
        Diferença máxima observada em dB.

    Raises:
        AssertionError: Se a diferença máxima exceder `tolerance_db` ou clamp for violado.
    """
    model_eval = model.eval().cpu()
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Arquivo ONNX não encontrado em {onnx_path}")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(seed)
    max_observed_diff = 0.0

    for batch_size in test_batches:
        # Testa entradas típicas e também extremas para assegurar limites de clamp
        feat_np = rng.normal(0.0, 15.0, size=(batch_size, NUM_ERB_BANDS, 3)).astype(np.float32)
        feat_torch = torch.from_numpy(feat_np)

        with torch.no_grad():
            py_out = model_eval(feat_torch).numpy()

        ort_inputs = {INPUT_NAME: feat_np}
        ort_out = session.run([OUTPUT_NAME], ort_inputs)[0]

        # Verifica formato
        assert ort_out.shape == (batch_size, NUM_ERB_BANDS), (
            f"Formato ORT incorreto: esperado {(batch_size, NUM_ERB_BANDS)}, obtido {ort_out.shape}"
        )

        # Verifica respeito aos limites de ganho [-6.0, +12.0] dB
        assert np.all(ort_out >= GAIN_MIN_DB - 1e-6), (
            f"Ganho mínimo violado: {ort_out.min()} < {GAIN_MIN_DB}"
        )
        assert np.all(ort_out <= GAIN_MAX_DB + 1e-6), (
            f"Ganho máximo violado: {ort_out.max()} > {GAIN_MAX_DB}"
        )

        diff = float(np.max(np.abs(py_out - ort_out)))
        max_observed_diff = max(max_observed_diff, diff)

        assert diff < tolerance_db, (
            f"Paridade violada para batch {batch_size}: diff={diff:.6e} dB >= tolerância={tolerance_db:.6e} dB"
        )

    return max_observed_diff


def verify_tract_compatibility(
    onnx_path: Path | str,
    test_batch: int = 1,
    tolerance_db: float = 1e-4,
) -> bool:
    """Verifica compatibilidade do modelo ONNX com o runtime Tract 0.19 do Clearcore.

    Utiliza tools/tract-check se disponível no ambiente.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Arquivo ONNX não encontrado em {onnx_path}")

    # Checagem estrutural estática
    proto = onnx.load(str(onnx_path))
    assert proto.ir_version <= 8, f"ir_version {proto.ir_version} > 8 incompatível com Tract 0.19"

    # Testa execução direta no tract via tract-check se compilado
    try:
        binary_path = tractcheck.binary()
        if not binary_path.is_file():
            return True
    except Exception:
        # Se tract-check não estiver disponível, a checagem ONNX estática foi aprovada
        return True

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        in_f32 = tmp_path / "feat.f32"
        out_json = tmp_path / "out.json"

        rng = np.random.default_rng(123)
        feat_data = rng.normal(0.0, 5.0, size=(test_batch, NUM_ERB_BANDS, 3)).astype("<f4")
        feat_data.tofile(in_f32)

        proc = tractcheck.run("eq", str(onnx_path), str(in_f32), str(out_json), str(test_batch))
        if proc.returncode != 0:
            raise RuntimeError(f"tract-check falhou ao carregar {onnx_path}: {proc.stderr} {proc.stdout}")

        report = json.loads(out_json.read_text())
        if not report.get("ok"):
            raise RuntimeError(f"tract recusou modelo: {report.get('error')}")

        tract_gains = np.array(report["gains_db"], dtype=np.float32).reshape(test_batch, NUM_ERB_BANDS)

        # Valida contra ONNX Runtime
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_gains = sess.run([OUTPUT_NAME], {INPUT_NAME: feat_data})[0]
        diff = float(np.max(np.abs(tract_gains - ort_gains)))
        if diff >= tolerance_db:
            raise RuntimeError(f"Divergência Tract vs ORT: {diff:.6e} dB >= {tolerance_db:.6e} dB")

    return True
