#!/usr/bin/env python3
"""Inferência do Neural EQ para calibração acústica de microfones."""

import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort

from neural_eq.constants import NUM_ERB_BANDS, ERB_CENTERS_HZ
from neural_eq.data import extract_erb_features_from_audio


def main():
    parser = argparse.ArgumentParser(description="Neural EQ Inference")
    parser.add_argument("--model", type=str, default="models/neural_eq.onnx", help="Caminho do modelo ONNX")
    parser.add_argument("--audio", type=str, required=True, help="Áudio de entrada (.wav) para calibrar")
    parser.add_argument("--sr", type=int, default=48000, help="Taxa de amostragem (padrão 48kHz)")
    args = parser.parse_args()

    feat = extract_erb_features_from_audio(args.audio, target_sr=args.sr)  # [32, 3]
    feat_batch = np.expand_dims(feat, axis=0).astype(np.float32)           # [1, 32, 3]

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    outputs = session.run(["gains_db"], {"feat": feat_batch})
    gains_db = outputs[0][0]  # [32]

    print("\n=== Neural EQ: Ganhos Estimados de Calibração (32 bandas ERB) ===")
    for i, (center_hz, gain) in enumerate(zip(ERB_CENTERS_HZ, gains_db)):
        bar = "#" * int(max(0, min(20, (gain + 6.0) * 1.1)))
        print(f"Banda {i:02d} ({center_hz:5.0f} Hz): {gain:+6.2f} dB | {bar}")


if __name__ == "__main__":
    main()
