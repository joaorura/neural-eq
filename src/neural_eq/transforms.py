"""Microphone acoustic distortion simulation and ERB frequency transformations for Neural EQ.

Implements synthetic microphone frequency response generation d(f) in dB over 32 ERB bands
and 481 FFT bins (48 kHz / N=960), distortion families (bass roll-off, high-cut, proximity,
resonance peaks/shelves, and identity neutrality), and ideal inverse target gains:
target_gains = clamp(-d, -6.0, 12.0) dB.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

# ERB band layout for 48 kHz / N=960 (481 frequency bins, 50 Hz per bin).
# Identical to crates/model/src/spectral_eq.rs and libdf (DF(48000, 960, 480, 32, 2)).
ERB_WIDTHS: tuple[int, ...] = (
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    5,
    5,
    7,
    7,
    8,
    10,
    12,
    13,
    15,
    18,
    20,
    24,
    28,
    31,
    37,
    42,
    50,
    56,
    67,
)

NUM_ERB_BANDS: int = len(ERB_WIDTHS)  # 32
NUM_FFT_BINS: int = sum(ERB_WIDTHS)  # 481
SAMPLE_RATE: int = 48_000
FFT_SIZE: int = 960
HOP_SIZE: int = 480
BIN_HZ: float = SAMPLE_RATE / FFT_SIZE  # 50.0 Hz

MIN_GAIN_DB: float = -6.0
MAX_GAIN_DB: float = 12.0


def get_erb_centers(widths: Sequence[int] = ERB_WIDTHS) -> np.ndarray:
    """Return center bin positions (0-indexed float) for each ERB band.

    Matching crates/model/src/spectral_eq.rs:
    center[k] = start + (width - 1.0) / 2.0.
    """
    centers = np.zeros(len(widths), dtype=np.float32)
    start = 0
    for k, w in enumerate(widths):
        centers[k] = float(start) + (float(w) - 1.0) / 2.0
        start += w
    return centers


def get_erb_center_frequencies(
    widths: Sequence[int] = ERB_WIDTHS,
    sr: int = SAMPLE_RATE,
    n_fft: int = FFT_SIZE,
) -> np.ndarray:
    """Return center frequencies in Hz for each ERB band."""
    bin_hz = float(sr) / float(n_fft)
    return (get_erb_centers(widths) * bin_hz).astype(np.float32)


def get_erb_interpolation_matrix(widths: Sequence[int] = ERB_WIDTHS) -> np.ndarray:
    """Return (481, 32) matrix M mapping 32 ERB band gains (in dB) to 481 FFT bins (in dB).

    Matches Rust `crates/model/src/spectral_eq.rs::bin_factors`:
    - Bins at or below centers[0] receive gains_db[0].
    - Bins at or above centers[-1] receive gains_db[-1].
    - Bins between centers are linearly interpolated in dB.
    """
    total = sum(widths)
    num_bands = len(widths)
    centers = get_erb_centers(widths)
    matrix = np.zeros((total, num_bands), dtype=np.float32)

    for bin_idx in range(total):
        pos = float(bin_idx)
        if pos <= centers[0]:
            matrix[bin_idx, 0] = 1.0
        elif pos >= centers[-1]:
            matrix[bin_idx, -1] = 1.0
        else:
            band = 0
            while band + 1 < num_bands and centers[band + 1] <= pos:
                band += 1
            low_c = centers[band]
            high_c = centers[band + 1]
            t = (pos - low_c) / (high_c - low_c)
            matrix[bin_idx, band] = 1.0 - t
            matrix[bin_idx, band + 1] = t

    return matrix


_INTERPOLATION_MATRIX: np.ndarray = get_erb_interpolation_matrix()


def bin_factors_from_erb(
    gains_erb_db: np.ndarray,
    widths: Sequence[int] = ERB_WIDTHS,
) -> np.ndarray:
    """Convert 32 ERB band gains in dB to 481 linear factors applied to spectrum bins.

    Matches Rust `spectral_eq::bin_factors` with bit-level floating-point precision:
    factors[i] = 10 ** (bin_gain_db[i] / 20.0).
    """
    gains = np.asarray(gains_erb_db, dtype=np.float32)
    if gains.shape[-1] != len(widths):
        raise ValueError(f"Expected {len(widths)} gains, got shape {gains.shape}")
    m = _INTERPOLATION_MATRIX if len(widths) == NUM_ERB_BANDS else get_erb_interpolation_matrix(widths)
    db = gains @ m.T
    return (10.0 ** (db / 20.0)).astype(np.float32)


@dataclass(frozen=True)
class DistortionCurve:
    """Synthetic microphone response distortion curve and ideal inverse target."""

    d_erb: np.ndarray  # shape [32], channel distortion in dB on ERB bands
    d_bins: np.ndarray  # shape [481], channel distortion in dB on FFT bins
    target_gains: np.ndarray  # shape [32], ideal inverse gains clamped to [-6.0, 12.0] dB
    family: str  # family name: 'identity', 'bass_rolloff', 'high_cut', etc.


class MicrophoneDistortionGenerator:
    """Generates synthetic microphone distortion curves across multiple acoustic families."""

    def __init__(
        self,
        identity_prob: float = 0.25,
        family_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.identity_prob = float(np.clip(identity_prob, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.centers_hz = get_erb_center_frequencies()
        self.interp_m = _INTERPOLATION_MATRIX

        # Default weights among non-identity distortion families
        default_weights = {
            "bass_rolloff": 0.25,
            "high_cut": 0.25,
            "proximity": 0.20,
            "resonance": 0.15,
            "composite": 0.15,
        }
        if family_weights is not None:
            default_weights.update(family_weights)

        # Normalize non-identity weights
        families = list(default_weights.keys())
        raw_w = np.array([default_weights[f] for f in families], dtype=np.float64)
        if raw_w.sum() <= 0:
            raw_w = np.ones(len(families), dtype=np.float64)
        self.family_names = families
        self.family_probs = raw_w / raw_w.sum()

    def sample_identity(self) -> DistortionCurve:
        """Neutral response (d = 0 dB across all bands, target = 0 dB)."""
        d_erb = np.zeros(NUM_ERB_BANDS, dtype=np.float32)
        d_bins = np.zeros(NUM_FFT_BINS, dtype=np.float32)
        target_gains = np.zeros(NUM_ERB_BANDS, dtype=np.float32)
        return DistortionCurve(d_erb=d_erb, d_bins=d_bins, target_gains=target_gains, family="identity")

    def sample_bass_rolloff(self, rng: np.random.Generator | None = None) -> DistortionCurve:
        """Low-frequency roll-off (80 - 300 Hz cutoff) simulating small mic capsules / high-pass."""
        gen = rng or self.rng
        fc = float(gen.uniform(80.0, 300.0))
        atten_db = float(gen.uniform(-12.0, -3.0))
        order = int(gen.choice([1, 2]))

        # Low-shelf / high-pass attenuation curve
        ratio = self.centers_hz / fc
        d_erb = atten_db / (1.0 + (ratio ** (2.0 * order)))
        d_erb = d_erb.astype(np.float32)

        d_bins = (self.interp_m @ d_erb).astype(np.float32)
        target_gains = np.clip(-d_erb, MIN_GAIN_DB, MAX_GAIN_DB).astype(np.float32)
        return DistortionCurve(d_erb=d_erb, d_bins=d_bins, target_gains=target_gains, family="bass_rolloff")

    def sample_high_cut(self, rng: np.random.Generator | None = None) -> DistortionCurve:
        """High-frequency attenuation (4 - 8 kHz cutoff) simulating muffled/cheap microphones."""
        gen = rng or self.rng
        fc = float(gen.uniform(4000.0, 8000.0))
        atten_db = float(gen.uniform(-12.0, -3.0))
        order = int(gen.choice([1, 2]))

        # High-shelf attenuation curve
        ratio = fc / np.maximum(self.centers_hz, 1.0)
        d_erb = atten_db / (1.0 + (ratio ** (2.0 * order)))
        d_erb = d_erb.astype(np.float32)

        d_bins = (self.interp_m @ d_erb).astype(np.float32)
        target_gains = np.clip(-d_erb, MIN_GAIN_DB, MAX_GAIN_DB).astype(np.float32)
        return DistortionCurve(d_erb=d_erb, d_bins=d_bins, target_gains=target_gains, family="high_cut")

    def sample_proximity(self, rng: np.random.Generator | None = None) -> DistortionCurve:
        """Proximity effect (+graves) boosting frequencies below 150 - 350 Hz."""
        gen = rng or self.rng
        fc = float(gen.uniform(150.0, 350.0))
        boost_db = float(gen.uniform(2.0, 8.5))

        # Low-shelf boost curve
        ratio = self.centers_hz / fc
        d_erb = boost_db / (1.0 + (ratio**2.0))
        d_erb = d_erb.astype(np.float32)

        d_bins = (self.interp_m @ d_erb).astype(np.float32)
        target_gains = np.clip(-d_erb, MIN_GAIN_DB, MAX_GAIN_DB).astype(np.float32)
        return DistortionCurve(d_erb=d_erb, d_bins=d_bins, target_gains=target_gains, family="proximity")

    def sample_resonance(self, rng: np.random.Generator | None = None) -> DistortionCurve:
        """Acoustic resonance (peaks/shelves) simulating housing or cavity resonances."""
        gen = rng or self.rng
        num_peaks = int(gen.integers(1, 4))
        d_erb = np.zeros(NUM_ERB_BANDS, dtype=np.float32)

        for _ in range(num_peaks):
            f0 = float(gen.uniform(300.0, 5000.0))
            # Sample gain avoiding near-zero
            gain_sign = float(gen.choice([-1.0, 1.0]))
            gain_mag = float(gen.uniform(2.0, 7.5))
            gain_db = gain_sign * gain_mag
            q = float(gen.uniform(0.8, 3.5))

            bandwidth = 1.0 / q
            delta_oct = np.log2(np.maximum(self.centers_hz, 1.0) / f0)
            peak = gain_db * np.exp(-0.5 * (delta_oct / (bandwidth / 2.0)) ** 2)
            d_erb += peak.astype(np.float32)

        # Bound distortion to maintain realistic range
        d_erb = np.clip(d_erb, -14.0, 8.0).astype(np.float32)
        d_bins = (self.interp_m @ d_erb).astype(np.float32)
        target_gains = np.clip(-d_erb, MIN_GAIN_DB, MAX_GAIN_DB).astype(np.float32)
        return DistortionCurve(d_erb=d_erb, d_bins=d_bins, target_gains=target_gains, family="resonance")

    def sample_composite(self, rng: np.random.Generator | None = None) -> DistortionCurve:
        """Composite curve combining multiple distortion families or smooth spectral variations."""
        gen = rng or self.rng
        mode = gen.choice(["combined_shelves", "cosine_series"])

        if mode == "combined_shelves":
            # Combine bass shaping (rolloff or proximity) with high-cut and presence
            d_erb = np.zeros(NUM_ERB_BANDS, dtype=np.float32)
            if gen.random() < 0.5:
                # Bass roll-off
                fc_b = float(gen.uniform(80.0, 250.0))
                att_b = float(gen.uniform(-9.0, -2.0))
                d_erb += att_b / (1.0 + (self.centers_hz / fc_b) ** 2.0)
            else:
                # Proximity boost
                fc_p = float(gen.uniform(150.0, 300.0))
                bst_p = float(gen.uniform(2.0, 6.0))
                d_erb += bst_p / (1.0 + (self.centers_hz / fc_p) ** 2.0)

            # High cut
            if gen.random() < 0.7:
                fc_h = float(gen.uniform(5000.0, 9000.0))
                att_h = float(gen.uniform(-9.0, -2.0))
                d_erb += att_h / (1.0 + (fc_h / np.maximum(self.centers_hz, 1.0)) ** 2.0)

            # Presence peak / dip in mid-range (1.5 - 4 kHz)
            if gen.random() < 0.6:
                f0 = float(gen.uniform(1500.0, 4000.0))
                g = float(gen.uniform(-4.0, 4.0))
                delta_oct = np.log2(np.maximum(self.centers_hz, 1.0) / f0)
                d_erb += g * np.exp(-0.5 * (delta_oct / 0.7) ** 2)
        else:
            # Smooth series of cosine basis functions on ERB band axis
            num_harmonics = int(gen.integers(2, 5))
            k_axis = np.arange(NUM_ERB_BANDS, dtype=np.float32) + 0.5
            d_erb = np.zeros(NUM_ERB_BANDS, dtype=np.float32)
            for j in range(1, num_harmonics + 1):
                amp = float(gen.uniform(-3.5, 3.5)) / j
                d_erb += amp * np.cos(np.pi * j * k_axis / NUM_ERB_BANDS)

        d_erb = np.clip(d_erb, -14.0, 8.0).astype(np.float32)
        d_bins = (self.interp_m @ d_erb).astype(np.float32)
        target_gains = np.clip(-d_erb, MIN_GAIN_DB, MAX_GAIN_DB).astype(np.float32)
        return DistortionCurve(d_erb=d_erb, d_bins=d_bins, target_gains=target_gains, family="composite")

    def sample_curve(
        self,
        family: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> DistortionCurve:
        """Sample a distortion curve, honoring the identity fraction and distortion families."""
        gen = rng or self.rng

        if family is not None:
            chosen = family
        elif gen.random() < self.identity_prob:
            chosen = "identity"
        else:
            chosen = str(gen.choice(self.family_names, p=self.family_probs))

        if chosen == "identity":
            return self.sample_identity()
        if chosen == "bass_rolloff":
            return self.sample_bass_rolloff(gen)
        if chosen == "high_cut":
            return self.sample_high_cut(gen)
        if chosen == "proximity":
            return self.sample_proximity(gen)
        if chosen == "resonance":
            return self.sample_resonance(gen)
        if chosen == "composite":
            return self.sample_composite(gen)

        raise ValueError(f"Unknown distortion family: {chosen}")


def design_fir_filter(d_bins_db: np.ndarray, n_taps: int = 961) -> np.ndarray:
    """Design a linear-phase FIR filter from 481 frequency bins (in dB).

    Uses type-I odd-length symmetric FIR filter design via `scipy.signal.firwin2`.
    """
    d_bins = np.asarray(d_bins_db, dtype=np.float32)
    if d_bins.size != NUM_FFT_BINS:
        raise ValueError(f"Expected {NUM_FFT_BINS} frequency bins, got {d_bins.size}")

    if np.allclose(d_bins, 0.0, atol=1e-5):
        taps = np.zeros(n_taps, dtype=np.float32)
        taps[n_taps // 2] = 1.0
        return taps

    # firwin2 requires normalized frequencies in [0, 1]
    freqs = np.linspace(0.0, 1.0, len(d_bins), dtype=np.float64)
    gains = (10.0 ** (d_bins / 20.0)).astype(np.float64)
    taps = signal.firwin2(n_taps, freqs, gains)
    return taps.astype(np.float32)


def apply_distortion(
    audio: np.ndarray,
    d_bins_db: np.ndarray,
    n_taps: int = 961,
) -> np.ndarray:
    """Apply the synthetic microphone distortion curve d(f) to time-domain audio."""
    audio_f = np.asarray(audio, dtype=np.float32)
    if audio_f.size == 0 or np.allclose(d_bins_db, 0.0, atol=1e-5):
        return audio_f.copy()

    taps = design_fir_filter(d_bins_db, n_taps=n_taps)
    filtered = signal.fftconvolve(audio_f, taps, mode="same")
    return filtered.astype(np.float32)
