"""PyTorch Dataset and ERB feature extraction for Neural EQ microphone compensation.

Extracts input features feat [32, 3] (mean, P10, P90 of log energy in dB per ERB band
over active speech frames after global level normalization) and valid mask [32]
according to the spec in docs/superpowers/research/2026-10-03-clearcore-train/08-eq-neural.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.signal as signal
import soundfile as sf
import soxr
import torch
from torch.utils.data import Dataset

from neural_eq.transforms import (
    ERB_WIDTHS,
    FFT_SIZE,
    HOP_SIZE,
    NUM_ERB_BANDS,
    SAMPLE_RATE,
    MicrophoneDistortionGenerator,
    apply_distortion,
    get_erb_center_frequencies,
)

# Optional libdf import for hardware-accelerated STFT/ERB analysis
try:
    from libdf import DF, erb

    _HAS_LIBDF = True
except ImportError:
    _HAS_LIBDF = False

_EPS: float = 1e-12


def _compute_erb_powers_numpy(
    audio_48k: np.ndarray,
    widths: Sequence[int] = ERB_WIDTHS,
    n_fft: int = FFT_SIZE,
    hop_length: int = HOP_SIZE,
) -> np.ndarray:
    """Compute ERB band log-energy in dB per frame using pure numpy/scipy STFT."""
    window = np.hanning(n_fft).astype(np.float32)
    # Pad audio to center frames like libdf
    pad = n_fft // 2
    padded = np.pad(audio_48k, (pad, pad), mode="reflect")

    # Framing
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    shape = (n_frames, n_fft)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Windowed FFT
    windowed = frames * window
    spec = np.fft.rfft(windowed, n=n_fft, axis=-1)  # shape [T, n_fft // 2 + 1]
    power = (spec.real**2 + spec.imag**2).astype(np.float32)

    # Average power per ERB band
    erb_db = np.zeros((n_frames, len(widths)), dtype=np.float32)
    start = 0
    for k, w in enumerate(widths):
        band_power = np.mean(power[:, start : start + w], axis=-1)
        erb_db[:, k] = 10.0 * np.log10(np.maximum(band_power, _EPS))
        start += w

    return erb_db


def _compute_erb_powers_libdf(
    audio_48k: np.ndarray,
    widths: Sequence[int] = ERB_WIDTHS,
) -> np.ndarray:
    """Compute ERB band log-energy in dB per frame using libdf."""
    df_state = DF(SAMPLE_RATE, FFT_SIZE, HOP_SIZE, len(widths), 2)
    in_arr = np.ascontiguousarray(audio_48k, dtype=np.float32)[None]
    spec = df_state.analysis(in_arr)
    # libdf erb returns shape [1, T, 32] in dB
    erb_db = erb(spec, df_state.erb_widths())[0]
    return np.asarray(erb_db, dtype=np.float32)


def extract_eq_features(
    audio_48k: np.ndarray,
    sr_input: int = SAMPLE_RATE,
    vad_db_below_peak: float = 30.0,
    abs_floor_dbfs: float = -70.0,
    use_libdf: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Neural EQ input features [32, 3] and valid mask [32].

    Args:
        audio_48k: 1D float32 numpy array sampled at 48 kHz.
        sr_input: Original sample rate of the recording before 48k resampling (e.g. 16000
            for enrollment audio). Bands above Nyquist (sr_input / 2) receive valid = 0.0.
        vad_db_below_peak: VAD energy dynamic range below peak active frame in dB.
        abs_floor_dbfs: Absolute floor below which frames are treated as inactive silence.
        use_libdf: Whether to use libdf if available.

    Returns:
        feat: np.ndarray of shape [32, 3] (dtype float32).
              Column 0: Mean log energy in dB per band over active speech frames (after global normalization).
              Column 1: 10th percentile (P10) log energy in dB.
              Column 2: 90th percentile (P90) log energy in dB.
        valid: np.ndarray of shape [32] (dtype float32).
               1.0 for observable ERB bands (center frequency <= sr_input / 2.0), 0.0 otherwise.
    """
    audio = np.asarray(audio_48k, dtype=np.float32).ravel()
    if audio.size == 0:
        feat = np.zeros((NUM_ERB_BANDS, 3), dtype=np.float32)
        valid = np.zeros(NUM_ERB_BANDS, dtype=np.float32)
        return feat, valid

    # 1. Compute per-frame ERB band log energy in dB [T, 32]
    if use_libdf and _HAS_LIBDF:
        erb_db = _compute_erb_powers_libdf(audio)
    else:
        erb_db = _compute_erb_powers_numpy(audio)

    # 2. VAD: frame-level broadband energy
    frame_pow = np.mean(10.0 ** (erb_db / 10.0), axis=-1)
    frame_db = 10.0 * np.log10(np.maximum(frame_pow, _EPS))
    peak_db = float(np.max(frame_db))

    mask = (frame_db >= peak_db - vad_db_below_peak) & (frame_db > abs_floor_dbfs)
    # If too few frames qualify (e.g. mostly silent clip), relax threshold
    if np.sum(mask) < 5:
        mask = frame_db >= (peak_db - 40.0)
    if np.sum(mask) < 1:
        mask = np.ones(len(frame_db), dtype=bool)

    active_erb = erb_db[mask]  # shape [N_active, 32]

    # 3. Global level normalization: normalize out overall loudness
    global_level = float(np.mean(active_erb))
    active_norm = active_erb - global_level

    # 4. Summary statistics per ERB band: mean, P10, P90
    mean_db = np.mean(active_norm, axis=0)
    p10_db = np.percentile(active_norm, 10.0, axis=0)
    p90_db = np.percentile(active_norm, 90.0, axis=0)
    feat = np.stack([mean_db, p10_db, p90_db], axis=-1).astype(np.float32)

    # 5. Observability valid mask based on Nyquist cutoff
    nyquist_hz = float(sr_input) / 2.0
    centers_hz = get_erb_center_frequencies()
    valid = (centers_hz <= nyquist_hz).astype(np.float32)

    return feat, valid


def generate_synthetic_speech(
    duration_sec: float = 6.0,
    sr: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate stochastic synthetic speech signal with realistic LTAS and formant structure."""
    gen = rng or np.random.default_rng()
    num_samples = int(duration_sec * sr)
    t = np.arange(num_samples, dtype=np.float64) / float(sr)

    # 1. Glottal source: F0 pitch oscillation with light vibrato
    f0 = float(gen.uniform(110.0, 240.0))
    vibrato_rate = float(gen.uniform(4.0, 6.0))
    vibrato_depth = float(gen.uniform(0.01, 0.03))
    f0_mod = f0 * (1.0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t))
    phase = 2.0 * np.pi * np.cumsum(f0_mod) / float(sr)

    # Sum of pitch harmonics with 1/n roll-off
    num_harmonics = min(40, int(8000.0 / f0))
    source = np.zeros(num_samples, dtype=np.float64)
    for n in range(1, num_harmonics + 1):
        amp = 1.0 / (n**0.85)
        source += amp * np.sin(n * phase + gen.uniform(0, 2.0 * np.pi))

    # 2. Add light breath / unvoiced noise component
    noise = gen.normal(0.0, 0.15, num_samples)
    sig = source + noise

    # 3. Formant resonant filtering (F1, F2, F3 vowel resonances)
    formants = [
        (float(gen.uniform(400.0, 800.0)), float(gen.uniform(3.0, 6.0))),
        (float(gen.uniform(1100.0, 2200.0)), float(gen.uniform(4.0, 8.0))),
        (float(gen.uniform(2500.0, 3400.0)), float(gen.uniform(5.0, 10.0))),
    ]
    filtered = np.zeros_like(sig)
    for f_center, q in formants:
        b, a = signal.iirpeak(f_center, q, fs=sr)
        filtered += signal.lfilter(b, a, sig)
    sig = sig * 0.2 + filtered * 0.8

    # 4. Syllabic / phrase amplitude envelope modulation (bursts and pauses)
    mod_freq = float(gen.uniform(2.5, 4.5))
    syllable_env = np.maximum(0.0, np.sin(2.0 * np.pi * mod_freq * t)) ** 1.5

    # Random intermittent pauses
    pause_points = gen.choice(num_samples, size=int(duration_sec * 0.8), replace=False)
    pause_mask = np.ones(num_samples, dtype=np.float64)
    pause_len = int(0.15 * sr)
    for p in pause_points:
        p_end = min(num_samples, p + pause_len)
        pause_mask[p:p_end] = 0.0

    sig = sig * syllable_env * pause_mask

    # 5. Normalize level to speech RMS (-22 dBFS)
    rms = np.sqrt(np.mean(sig**2) + 1e-12)
    target_rms = 10.0 ** (-22.0 / 20.0)
    sig = sig * (target_rms / (rms + 1e-6))
    return np.clip(sig, -0.99, 0.99).astype(np.float32)


def load_speech_audio(path: Path | str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load clean speech audio from .npy, .wav, or .flac file, normalized to float32 at target_sr."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        if arr.dtype == np.int16:
            audio = arr.astype(np.float32) / 32768.0
        else:
            audio = arr.astype(np.float32)
        return audio

    data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        mono = soxr.resample(mono, sr, target_sr, quality="HQ").astype(np.float32)
    return mono


def scan_clean_audio_files(data_dir: Path | str | None) -> list[Path]:
    """Scan data directory for clean speech recordings."""
    if data_dir is None:
        return []
    root = Path(data_dir)
    if not root.exists():
        return []

    files: list[Path] = []
    # Search for clean npy files first
    files.extend(root.glob("**/*.clean.npy"))
    if not files:
        # Fall back to wav/flac audio files, ignoring noise/rir folders
        for ext in ("*.wav", "*.flac"):
            for f in root.glob(f"**/{ext}"):
                parts = [p.lower() for p in f.parts]
                if not any(skip in parts for skip in ("noise", "rir", "interferer")):
                    files.append(f)
    return sorted(files)


class NeuralEqDataset(Dataset):
    """PyTorch Dataset for training Neural EQ microphone compensation models.

    Loads clean speech audio from data/ (or generates synthetic speech on the fly),
    applies synthetic microphone distortion d(f), and extracts feature tensors
    feat [32, 3] and valid [32] alongside ideal inverse target gains [32].
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        file_list: Sequence[Path | str] | None = None,
        synthetic_fallback: bool = True,
        sample_rate: int = SAMPLE_RATE,
        duration_sec: float = 6.0,
        length: int | None = None,
        distortion_generator: MicrophoneDistortionGenerator | None = None,
        identity_prob: float = 0.25,
        sr_input_prob_16k: float = 0.0,
        random_crop: bool = True,
        return_audio: bool = False,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.target_samples = int(duration_sec * sample_rate)
        self.random_crop = random_crop
        self.return_audio = return_audio
        self.sr_input_prob_16k = float(np.clip(sr_input_prob_16k, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

        # Distortion generator
        if distortion_generator is not None:
            self.dist_gen = distortion_generator
        else:
            self.dist_gen = MicrophoneDistortionGenerator(identity_prob=identity_prob, seed=seed)

        # Audio files discovery
        self.files: list[Path] = []
        if file_list is not None:
            self.files = [Path(f) for f in file_list]
        elif data_dir is not None:
            self.files = scan_clean_audio_files(data_dir)

        self.synthetic_mode = len(self.files) == 0
        if self.synthetic_mode and not synthetic_fallback:
            raise FileNotFoundError(f"No clean audio files found in {data_dir} and synthetic_fallback is False")

        # Dataset epoch length
        if length is not None:
            self.length = int(length)
        elif self.synthetic_mode:
            self.length = 1000
        else:
            self.length = len(self.files)

    def __len__(self) -> int:
        return self.length

    def _get_clean_audio(self, idx: int) -> np.ndarray:
        if self.synthetic_mode:
            return generate_synthetic_speech(duration_sec=self.duration_sec, sr=self.sample_rate, rng=self.rng)

        # Pick audio file
        file_idx = idx % len(self.files)
        path = self.files[file_idx]
        try:
            audio = load_speech_audio(path, target_sr=self.sample_rate)
        except Exception:
            # Fall back to synthetic on read failure
            return generate_synthetic_speech(duration_sec=self.duration_sec, sr=self.sample_rate, rng=self.rng)

        if audio.size == 0:
            return generate_synthetic_speech(duration_sec=self.duration_sec, sr=self.sample_rate, rng=self.rng)

        # Slice or pad to target length
        if len(audio) >= self.target_samples:
            if self.random_crop:
                max_start = len(audio) - self.target_samples
                start = int(self.rng.integers(0, max_start + 1))
            else:
                start = (len(audio) - self.target_samples) // 2
            return audio[start : start + self.target_samples]

        # Pad if shorter
        pad_len = self.target_samples - len(audio)
        return np.pad(audio, (0, pad_len), mode="reflect")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        # 1. Obtain clean speech audio
        clean_audio = self._get_clean_audio(idx)

        # 2. Sample synthetic microphone distortion curve
        curve = self.dist_gen.sample_curve(rng=self.rng)

        # 3. Apply distortion to audio in time domain
        distorted_audio = apply_distortion(clean_audio, curve.d_bins)

        # 4. Determine input recording sample rate (simulates 16k enrollment vs 48k studio)
        sr_in = 16000 if (self.rng.random() < self.sr_input_prob_16k) else 48000

        # 5. Extract input features feat [32, 3] and valid [32]
        feat, valid = extract_eq_features(distorted_audio, sr_input=sr_in)

        item: dict[str, torch.Tensor | str] = {
            "feat": torch.from_numpy(feat).float(),
            "valid": torch.from_numpy(valid).float(),
            "target_gains": torch.from_numpy(curve.target_gains).float(),
            "d_erb": torch.from_numpy(curve.d_erb).float(),
            "d_bins": torch.from_numpy(curve.d_bins).float(),
            "family": curve.family,
        }

        if self.return_audio:
            item["clean_audio"] = torch.from_numpy(clean_audio).float()
            item["distorted_audio"] = torch.from_numpy(distorted_audio).float()

        return item
