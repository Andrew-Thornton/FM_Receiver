import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import sys

# =============================================================================
# Configuration
# =============================================================================

FILE_PATH = "./raw_data_files/decimated_data_240kHz.bin"

INPUT_RATE = 240_000      # IQ sample rate
AUDIO_RATE = 48_000       # output audio sample rate

DEEMPHASIS_US = 50        # Australia = 50 µs

# =============================================================================
# Load IQ
# =============================================================================

if not os.path.exists(FILE_PATH):
    print(f"ERROR: File not found: {FILE_PATH}")
    sys.exit(1)

file_bytes = os.path.getsize(FILE_PATH)

raw = np.fromfile(
    FILE_PATH,
    dtype=np.int16,
    count=(file_bytes // 2)
)

iq = (
    raw[0::2].astype(np.float32)
    + 1j * raw[1::2].astype(np.float32)
)

print(f"Loaded {len(iq):,} IQ samples")
print(f"Sample rate = {INPUT_RATE/1e3:.1f} kHz")


# =============================================================================
# Filter helpers
# =============================================================================

def butter_lowpass(fc, fs, order=6):
    return signal.butter(order, fc, btype="low", fs=fs)

def butter_bandpass(f1, f2, fs, order=4):
    return signal.butter(order, [f1, f2], btype="band", fs=fs)

def lowpass_filter(x, fc, fs, order=6):
    b, a = butter_lowpass(fc, fs, order)
    return signal.filtfilt(b, a, x)

def bandpass_filter(x, f1, f2, fs, order=4):
    b, a = butter_bandpass(f1, f2, fs, order)
    return signal.filtfilt(b, a, x)


# =============================================================================
# FM Demodulation
# =============================================================================
#
# angle(x[n] * conj(x[n-1])) gives phase delta/sample
#

print("FM demodulating...")

fm_demod = np.angle(iq[1:] * np.conj(iq[:-1]))

# Scale to Hz-ish units
fm_demod *= INPUT_RATE / (2 * np.pi)

time = np.arange(len(fm_demod)) / INPUT_RATE


# =============================================================================
# Stereo Decode
# =============================================================================

print("Extracting L+R audio...")

# Mono channel (0–15 kHz)
sum_lr = lowpass_filter(
    fm_demod,
    fc=15e3,
    fs=INPUT_RATE
)

print("Recovering 19 kHz pilot...")

pilot = bandpass_filter(
    fm_demod,
    f1=18.5e3,
    f2=19.5e3,
    fs=INPUT_RATE
)

# =============================================================================
# Recover coherent 38 kHz carrier
# =============================================================================
#
# Use Hilbert transform to estimate pilot phase,
# then double phase → coherent 38 kHz reference.
#

analytic_pilot = signal.hilbert(pilot)

pilot_phase = np.unwrap(
    np.angle(analytic_pilot)
)

carrier_38k = np.cos(
    2.0 * pilot_phase
)

print("Extracting stereo difference signal (L-R)...")

# Stereo DSB-SC band
lr_band = bandpass_filter(
    fm_demod,
    f1=23e3,
    f2=53e3,
    fs=INPUT_RATE
)

# Synchronous demodulation
diff_mixed = lr_band * carrier_38k * 2.0

# Lowpass back to audio
diff_lr = lowpass_filter(
    diff_mixed,
    fc=15e3,
    fs=INPUT_RATE
)

# =============================================================================
# Reconstruct Left / Right
# =============================================================================

print("Reconstructing stereo channels...")

left = 0.5 * (sum_lr + diff_lr)
right = 0.5 * (sum_lr - diff_lr)


# =============================================================================
# De-emphasis filter
# =============================================================================

print(f"Applying {DEEMPHASIS_US} µs deemphasis...")

tau = DEEMPHASIS_US * 1e-6

alpha = 1.0 / (1.0 + INPUT_RATE * tau)

def deemphasis(x):
    y = np.zeros_like(x)

    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (
            x[i] - y[i - 1]
        )

    return y

left = deemphasis(left)
right = deemphasis(right)


# =============================================================================
# Resample to audio rate
# =============================================================================

print("Resampling audio...")

num_output = int(
    len(left) * AUDIO_RATE / INPUT_RATE
)

left_audio = signal.resample(
    left,
    num_output
)

right_audio = signal.resample(
    right,
    num_output
)


# =============================================================================
# Normalize audio
# =============================================================================

peak = max(
    np.max(np.abs(left_audio)),
    np.max(np.abs(right_audio))
)

left_audio /= (peak + 1e-12)
right_audio /= (peak + 1e-12)


# =============================================================================
# Save WAV
# =============================================================================

stereo_audio = np.stack(
    [left_audio, right_audio],
    axis=1
)

stereo_audio_int16 = np.int16(
    stereo_audio * 32767
)

# =============================================================================
# Save WAVs
# =============================================================================

print("Saving WAV files...")

os.makedirs("./output_files", exist_ok=True)

# -------------------------------------------------------------------------
# Mono output (L+R)
# -------------------------------------------------------------------------

mono_audio = signal.resample(
    sum_lr,
    num_output
)

mono_audio /= (
    np.max(np.abs(mono_audio)) + 1e-12
)

mono_audio_int16 = np.int16(
    mono_audio * 32767
)

mono_path = "./output_files/mono_out.wav"

wavfile.write(
    mono_path,
    AUDIO_RATE,
    mono_audio_int16
)

print(f"Saved mono WAV: {mono_path}")

# -------------------------------------------------------------------------
# Stereo output (Left / Right)
# -------------------------------------------------------------------------

stereo_audio = np.stack(
    [left_audio, right_audio],
    axis=1
)

stereo_audio_int16 = np.int16(
    stereo_audio * 32767
)

stereo_path = "./output_files/stereo_out.wav"

wavfile.write(
    stereo_path,
    AUDIO_RATE,
    stereo_audio_int16
)

print(f"Saved stereo WAV: {stereo_path}")


# =============================================================================
# Diagnostics / Plots
# =============================================================================

print("Generating plots...")

fig, axes = plt.subplots(
    5,
    1,
    figsize=(14, 14)
)

fig.suptitle(
    "FM Stereo Decode",
    fontsize=14,
    fontweight="bold"
)

# -------------------------------------------------------------------------
# 1. MPX signal
# -------------------------------------------------------------------------

axes[0].plot(
    time[:5000] * 1e3,
    fm_demod[:5000],
    linewidth=0.7
)

axes[0].set_title(
    "FM Demodulated MPX Signal"
)

axes[0].set_xlabel("Time (ms)")
axes[0].grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# 2. MPX spectrum
# -------------------------------------------------------------------------

window = np.hanning(len(fm_demod))

fft = np.fft.rfft(
    fm_demod * window
)

freqs = np.fft.rfftfreq(
    len(fm_demod),
    d=1 / INPUT_RATE
)

mag_db = 20 * np.log10(
    np.abs(fft) + 1e-12
)

mag_db -= mag_db.max()

axes[1].plot(
    freqs / 1e3,
    mag_db,
    linewidth=0.8
)

axes[1].set_xlim(0, 80)
axes[1].set_ylim(-100, 5)

axes[1].axvline(
    19,
    linestyle="--"
)

axes[1].axvline(
    38,
    linestyle="--"
)

axes[1].set_title(
    "FM Multiplex Spectrum"
)

axes[1].set_xlabel("Frequency (kHz)")
axes[1].set_ylabel("Magnitude (dBr)")
axes[1].grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# 3. L+R
# -------------------------------------------------------------------------

axes[2].plot(
    time[:5000] * 1e3,
    sum_lr[:5000]
)

axes[2].set_title("L + R")
axes[2].set_xlabel("Time (ms)")
axes[2].grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# 4. L-R
# -------------------------------------------------------------------------

axes[3].plot(
    time[:5000] * 1e3,
    diff_lr[:5000]
)

axes[3].set_title("L - R")
axes[3].set_xlabel("Time (ms)")
axes[3].grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# 5. Stereo output
# -------------------------------------------------------------------------

axes[4].plot(
    left_audio[:5000],
    label="Left",
    linewidth=0.8
)

axes[4].plot(
    right_audio[:5000],
    label="Right",
    linewidth=0.8,
    alpha=0.8
)

axes[4].set_title(
    "Recovered Stereo Audio"
)

axes[4].legend()
axes[4].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "fm_stereo_decode.png",
    dpi=150
)

plt.show()

print("Saved: fm_stereo_decode.png")
print("Done.")