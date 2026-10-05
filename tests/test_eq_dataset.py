"""Unit tests for Neural EQ dataset, feature extraction, and microphone distortion simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from neural_eq.dataset import (
    NeuralEqDataset,
    extract_eq_features,
    generate_synthetic_speech,
    load_speech_audio,
)
from neural_eq.transforms import (
    ERB_WIDTHS,
    FFT_SIZE,
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    NUM_ERB_BANDS,
    NUM_FFT_BINS,
    SAMPLE_RATE,
    MicrophoneDistortionGenerator,
    apply_distortion,
    bin_factors_from_erb,
    design_fir_filter,
    get_erb_center_frequencies,
    get_erb_centers,
    get_erb_interpolation_matrix,
)


def test_erb_geometry_and_frequencies():
    """Verify 32 ERB bands, 481 FFT bins, center bins, and frequency values."""
    assert len(ERB_WIDTHS) == NUM_ERB_BANDS == 32
    assert sum(ERB_WIDTHS) == NUM_FFT_BINS == 481
    assert FFT_SIZE == 960
    assert SAMPLE_RATE == 48000

    centers = get_erb_centers()
    assert len(centers) == 32
    assert centers[0] == 0.5  # Band 0 has width 2: 0 + (2-1)/2 = 0.5
    assert centers[-1] == 447.0  # Band 31 starts at 414, width 67 -> 414 + 33 = 447

    centers_hz = get_erb_center_frequencies()
    assert len(centers_hz) == 32
    assert centers_hz[0] == 25.0  # 0.5 * 50 Hz
    assert centers_hz[24] == 7875.0  # Band 24 center (< 8 kHz)
    assert centers_hz[25] == 9175.0  # Band 25 center (> 8.5 kHz)
    assert centers_hz[-1] == 22350.0


def test_erb_interpolation_matrix_and_rust_parity():
    """Verify M matrix (481, 32) properties matching crates/model/src/spectral_eq.rs."""
    m = get_erb_interpolation_matrix()
    assert m.shape == (481, 32)
    assert m.dtype == np.float32

    # 1. Row sum equals 1.0 (convex linear combination)
    row_sums = m.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    # 2. All weights are non-negative
    assert np.all(m >= 0.0)

    # 3. Neutral gains (0 dB) yield exactly factor 1.0 on all 481 bins
    neutral_gains = np.zeros(32, dtype=np.float32)
    factors_neutral = bin_factors_from_erb(neutral_gains)
    assert len(factors_neutral) == 481
    np.testing.assert_allclose(factors_neutral, 1.0, atol=1e-6)

    # 4. Flat gain (+6 dB) gives 10**(6/20) on every bin
    flat_gains = np.full(32, 6.0, dtype=np.float32)
    factors_flat = bin_factors_from_erb(flat_gains)
    expected_factor = 10.0 ** (6.0 / 20.0)
    np.testing.assert_allclose(factors_flat, expected_factor, atol=1e-6)

    # 5. Odd-width band center bins preserve band gain exactly
    odd_gains = np.array([(-3.0 if k % 2 == 0 else 5.0) for k in range(32)], dtype=np.float32)
    factors_odd = bin_factors_from_erb(odd_gains)
    db_odd = 20.0 * np.log10(factors_odd)

    centers = get_erb_centers()
    for k, w in enumerate(ERB_WIDTHS):
        if w % 2 == 1:
            center_bin = int(round(centers[k]))
            assert pytest.approx(db_odd[center_bin], abs=1e-4) == odd_gains[k]

    # 6. Step between bands is smoothed and monotonic
    step_gains = np.array([0.0 if k < 16 else 12.0 for k in range(32)], dtype=np.float32)
    factors_step = bin_factors_from_erb(step_gains)
    db_step = 20.0 * np.log10(factors_step)
    assert np.all(np.diff(db_step) >= -1e-5)
    max_step = np.max(np.diff(db_step))
    assert max_step < 12.0  # Smooth transition without cliff


def test_distortion_families_and_target_gains():
    """Verify distortion generation across all families and target gain clipping."""
    gen = MicrophoneDistortionGenerator(seed=42)

    # 1. Identity
    c_id = gen.sample_identity()
    assert c_id.family == "identity"
    np.testing.assert_allclose(c_id.d_erb, 0.0)
    np.testing.assert_allclose(c_id.d_bins, 0.0)
    np.testing.assert_allclose(c_id.target_gains, 0.0)

    # 2. Bass roll-off
    c_bass = gen.sample_bass_rolloff()
    assert c_bass.family == "bass_rolloff"
    assert c_bass.d_erb[0] < -2.0  # Attenuation at 25 Hz
    assert abs(c_bass.d_erb[-1]) < 0.5  # High frequencies unaffected
    np.testing.assert_allclose(c_bass.target_gains, np.clip(-c_bass.d_erb, MIN_GAIN_DB, MAX_GAIN_DB))

    # 3. High cut (muffled microphone)
    c_high = gen.sample_high_cut()
    assert c_high.family == "high_cut"
    assert abs(c_high.d_erb[0]) < 0.5  # Low frequencies unaffected
    assert c_high.d_erb[-1] < -2.0  # High frequencies attenuated
    np.testing.assert_allclose(c_high.target_gains, np.clip(-c_high.d_erb, MIN_GAIN_DB, MAX_GAIN_DB))

    # 4. Proximity effect
    c_prox = gen.sample_proximity()
    assert c_prox.family == "proximity"
    assert c_prox.d_erb[0] > 1.5  # Bass boost
    assert abs(c_prox.d_erb[-1]) < 0.5  # Highs unaffected
    np.testing.assert_allclose(c_prox.target_gains, np.clip(-c_prox.d_erb, MIN_GAIN_DB, MAX_GAIN_DB))

    # 5. Resonance
    c_res = gen.sample_resonance()
    assert c_res.family == "resonance"
    assert np.any(np.abs(c_res.d_erb) > 1.0)
    np.testing.assert_allclose(c_res.target_gains, np.clip(-c_res.d_erb, MIN_GAIN_DB, MAX_GAIN_DB))

    # 6. Composite
    c_comp = gen.sample_composite()
    assert c_comp.family == "composite"
    assert len(c_comp.d_erb) == 32
    assert len(c_comp.d_bins) == 481
    np.testing.assert_allclose(c_comp.target_gains, np.clip(-c_comp.d_erb, MIN_GAIN_DB, MAX_GAIN_DB))

    # Strict bounds across all sampled families
    for curve in [c_id, c_bass, c_high, c_prox, c_res, c_comp]:
        assert np.all(curve.target_gains >= MIN_GAIN_DB)
        assert np.all(curve.target_gains <= MAX_GAIN_DB)


def test_distortion_generator_identity_fraction():
    """Verify that identity neutrality fraction (~25%) is respected statistically."""
    gen = MicrophoneDistortionGenerator(identity_prob=0.25, seed=123)
    n_samples = 1000
    families = [gen.sample_curve().family for _ in range(n_samples)]
    id_count = families.count("identity")
    id_fraction = id_count / n_samples
    assert 0.18 <= id_fraction <= 0.32

    # Fixed probabilities
    gen_pure_id = MicrophoneDistortionGenerator(identity_prob=1.0, seed=42)
    assert all(gen_pure_id.sample_curve().family == "identity" for _ in range(20))

    gen_no_id = MicrophoneDistortionGenerator(identity_prob=0.0, seed=42)
    assert all(gen_no_id.sample_curve().family != "identity" for _ in range(20))


def test_fir_design_and_apply_distortion():
    """Test FIR design and time-domain filtering with synthetic distortion curves."""
    audio = np.random.randn(48000).astype(np.float32)

    # Directly verify design_fir_filter
    taps_id = design_fir_filter(np.zeros(481, dtype=np.float32), n_taps=961)
    assert len(taps_id) == 961
    assert taps_id[480] == 1.0

    # Identity filtering produces identical audio
    d_zero = np.zeros(481, dtype=np.float32)
    out_ident = apply_distortion(audio, d_zero)
    np.testing.assert_allclose(audio, out_ident, atol=1e-5)

    # Bass roll-off attenuates a 100 Hz tone much more than a 2 kHz tone
    t = np.arange(48000) / 48000.0
    tone_low = np.sin(2.0 * np.pi * 100.0 * t).astype(np.float32)
    tone_mid = np.sin(2.0 * np.pi * 2000.0 * t).astype(np.float32)

    gen = MicrophoneDistortionGenerator(seed=42)
    c_bass = gen.sample_bass_rolloff()

    out_low = apply_distortion(tone_low, c_bass.d_bins)
    out_mid = apply_distortion(tone_mid, c_bass.d_bins)

    rms_low_in = np.sqrt(np.mean(tone_low[2000:-2000] ** 2))
    rms_low_out = np.sqrt(np.mean(out_low[2000:-2000] ** 2))
    rms_mid_in = np.sqrt(np.mean(tone_mid[2000:-2000] ** 2))
    rms_mid_out = np.sqrt(np.mean(out_mid[2000:-2000] ** 2))

    atten_low_db = 20.0 * np.log10(rms_low_out / rms_low_in)
    atten_mid_db = 20.0 * np.log10(rms_mid_out / rms_mid_in)

    assert atten_low_db < atten_mid_db - 2.0


def test_extract_eq_features_shapes_and_values():
    """Verify feature extraction contract: feat [32, 3] (mean, P10, P90) and valid [32]."""
    speech = generate_synthetic_speech(duration_sec=3.0, sr=SAMPLE_RATE, rng=np.random.default_rng(42))
    assert len(speech) == 3 * SAMPLE_RATE

    feat, valid = extract_eq_features(speech, sr_input=SAMPLE_RATE)
    assert feat.shape == (32, 3)
    assert feat.dtype == np.float32
    assert valid.shape == (32,)
    assert valid.dtype == np.float32
    assert np.all(np.isfinite(feat))

    # P10 <= P90 must hold for every band
    p10 = feat[:, 1]
    p90 = feat[:, 2]
    assert np.all(p10 <= p90 + 1e-4)

    # 48 kHz recording: all 32 bands are observable
    assert np.all(valid == 1.0)


def test_observability_valid_mask_at_16k():
    """Verify 16 kHz enrollment mask: bands 0..24 valid (<= 8 kHz), bands 25..31 invalid (> 8 kHz)."""
    speech = generate_synthetic_speech(duration_sec=2.0, sr=SAMPLE_RATE)
    _, valid_16k = extract_eq_features(speech, sr_input=16000)

    assert len(valid_16k) == 32
    assert int(valid_16k.sum()) == 25
    assert np.all(valid_16k[:25] == 1.0)
    assert np.all(valid_16k[25:] == 0.0)


def test_synthetic_speech_generator():
    """Verify synthetic speech generation properties (length, finite values, reasonable RMS)."""
    gen = np.random.default_rng(999)
    wav = generate_synthetic_speech(duration_sec=2.5, sr=48000, rng=gen)
    assert len(wav) == int(2.5 * 48000)
    assert wav.dtype == np.float32
    assert np.all(np.isfinite(wav))

    rms = np.sqrt(np.mean(wav**2))
    rms_db = 20.0 * np.log10(rms)
    assert -28.0 <= rms_db <= -16.0  # Typical speech level


def test_neural_eq_dataset_synthetic_mode():
    """Verify NeuralEqDataset in synthetic fallback mode with PyTorch DataLoader."""
    ds = NeuralEqDataset(
        data_dir=None,
        synthetic_fallback=True,
        duration_sec=2.0,
        length=8,
        seed=42,
    )
    assert len(ds) == 8

    item = ds[0]
    assert isinstance(item["feat"], torch.Tensor)
    assert item["feat"].shape == (32, 3)
    assert item["feat"].dtype == torch.float32

    assert isinstance(item["valid"], torch.Tensor)
    assert item["valid"].shape == (32,)
    assert item["valid"].dtype == torch.float32

    assert isinstance(item["target_gains"], torch.Tensor)
    assert item["target_gains"].shape == (32,)
    assert item["target_gains"].dtype == torch.float32
    assert item["target_gains"].min() >= MIN_GAIN_DB
    assert item["target_gains"].max() <= MAX_GAIN_DB

    assert isinstance(item["d_erb"], torch.Tensor)
    assert item["d_erb"].shape == (32,)

    assert isinstance(item["d_bins"], torch.Tensor)
    assert item["d_bins"].shape == (481,)

    assert isinstance(item["family"], str)

    # Test batching via DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["feat"].shape == (4, 32, 3)
    assert batch["valid"].shape == (4, 32)
    assert batch["target_gains"].shape == (4, 32)
    assert batch["d_erb"].shape == (4, 32)
    assert batch["d_bins"].shape == (4, 481)
    assert len(batch["family"]) == 4


def test_neural_eq_dataset_with_real_audio_if_present():
    """Verify NeuralEqDataset loading real clean audio files from data/dev-m2/."""
    dev_audio_dir = Path("data/dev-m2/dev-select/audio")
    if not dev_audio_dir.exists():
        pytest.skip("data/dev-m2/dev-select/audio not present on disk")

    ds = NeuralEqDataset(
        data_dir=dev_audio_dir,
        duration_sec=3.0,
        length=6,
        return_audio=True,
        seed=101,
    )
    assert not ds.synthetic_mode
    assert len(ds) == 6

    sample = ds[0]
    assert sample["feat"].shape == (32, 3)
    assert sample["valid"].shape == (32,)
    assert sample["target_gains"].shape == (32,)
    assert sample["clean_audio"].shape == (int(3.0 * SAMPLE_RATE),)
    assert sample["distorted_audio"].shape == (int(3.0 * SAMPLE_RATE),)


def test_audio_loader_with_npy_file(tmp_path: Path):
    """Test loading clean speech audio from numpy file."""
    fake_audio = (np.random.randn(96000) * 0.1).astype(np.float32)
    npy_path = tmp_path / "test.clean.npy"
    np.save(npy_path, (fake_audio * 32767.0).astype(np.int16))

    loaded = load_speech_audio(npy_path)
    assert loaded.dtype == np.float32
    assert len(loaded) == 96000
    np.testing.assert_allclose(loaded, fake_audio, atol=1e-3)
