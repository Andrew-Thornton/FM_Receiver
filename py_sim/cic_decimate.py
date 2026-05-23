"""
CIC Decimation Pipeline for IQ Binary Data
==========================================
Input:  raw_data_files/iq_245M.bin
Format: interleaved int16 [real, imag, real, imag, ...]
Rate:   245.76 MHz
Filter: 12x CIC decimate-by-2 = total /4096
Output: 245.76e6 / 4096 = 60 kHz effective sample rate
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ─── Configuration ────────────────────────────────────────────────────────────
FILE_PATH   = "raw_data_files/iq_245M.bin"
INPUT_RATE  = 245.76e6          # Hz
CIC_STAGES  = 11                # number of CIC filters
DECIMATION  = 2                 # each CIC decimates by this
TOTAL_DEC   = DECIMATION ** CIC_STAGES   # 2048
OUTPUT_RATE = INPUT_RATE / TOTAL_DEC     # ~60 kHz
MAX_SAMPLES = 1e99       # cap read at ~67M samples to avoid OOM

# ─── CIC Decimate-by-2 (integrator + comb + downsample) ───────────────────────
def cic_decimate_by2(iq: np.ndarray) -> np.ndarray:
    """
    Single CIC stage: decimate by 2.
      • Integrator  : running sum (IIR y[n] = x[n] + y[n-1])
      • Downsample  : keep every 2nd sample
      • Comb        : first-order difference (y[n] = x[n] - x[n-1])
    Applied independently to real and imaginary parts.
    """
    # Use int64 to avoid overflow accumulation across many stages
    r = iq.real.astype(np.int64)
    i = iq.imag.astype(np.int64)

    # Integrator
    r = np.cumsum(r)
    i = np.cumsum(i)

    # Downsample
    r = r[1::2]
    i = i[1::2]

    # Comb (first difference)
    r = np.diff(r, prepend=0)
    i = np.diff(i, prepend=0)

    return r.astype(np.float64) + 1j * i.astype(np.float64)


# ─── Load IQ file ─────────────────────────────────────────────────────────────
if not os.path.exists(FILE_PATH):
    print(f"ERROR: File not found: {FILE_PATH}")
    print("Place iq_245M.bin inside a 'raw_data_files/' directory next to this script.")
    sys.exit(1)

file_bytes = os.path.getsize(FILE_PATH)
max_bytes  = MAX_SAMPLES * 4   # 4 bytes per IQ pair (2× int16)
read_bytes = min(file_bytes, max_bytes)

print(f"File size : {file_bytes/1e6:.1f} MB")
print(f"Reading   : {read_bytes/1e6:.1f} MB  ({read_bytes//4:,} IQ samples)")

raw = np.fromfile(FILE_PATH, dtype=np.int16, count=(read_bytes // 2))
# Deinterleave: even indices = real, odd indices = imag
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
print(f"Loaded {len(iq):,} IQ samples @ {INPUT_RATE/1e6:.2f} MHz")

# ─── Apply 12 CIC stages ──────────────────────────────────────────────────────
signal = iq
for stage in range(1, CIC_STAGES + 1):
    signal = cic_decimate_by2(signal)
    rate_now = INPUT_RATE / (DECIMATION ** stage)
    print(f"  Stage {stage:2d}: {len(signal):>10,} samples  @ {rate_now/1e3:.2f} kHz")

print(f"\nFinal output: {len(signal):,} samples @ {OUTPUT_RATE/1e3:.3f} kHz")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle(
    f"CIC Decimation  ×4096  |  {INPUT_RATE/1e6:.2f} MHz  →  {OUTPUT_RATE/1e3:.2f} kHz",
    fontsize=13, fontweight="bold"
)

t = np.arange(len(signal)) / OUTPUT_RATE * 1e3   # time in ms

# — Time-domain: Real ---
ax = axes[0]
ax.plot(t, signal.real, color="#2196F3", linewidth=0.6, label="I (real)")
ax.plot(t, signal.imag, color="#FF9800", linewidth=0.6, alpha=0.75, label="Q (imag)")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude (counts)")
ax.set_title("Time Domain — I/Q after CIC decimation")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# — Power Spectrum (Welch) ---
from scipy.signal import welch
nperseg = min(1024, len(signal) // 4)
freq_r, psd_r = welch(signal.real, fs=OUTPUT_RATE, nperseg=nperseg)
freq_i, psd_i = welch(signal.imag, fs=OUTPUT_RATE, nperseg=nperseg)
freq_khz = freq_r / 1e3

ax = axes[1]
ax.semilogy(freq_khz, psd_r, color="#2196F3", linewidth=0.9, label="I PSD")
ax.semilogy(freq_khz, psd_i, color="#FF9800", linewidth=0.9, alpha=0.8, label="Q PSD")
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("PSD (counts²/Hz)")
ax.set_title("Power Spectral Density (Welch)")
ax.legend(loc="upper right")
ax.grid(True, which="both", alpha=0.3)

# — Complex spectrum (FFT) ---
N = len(signal)
window = np.hanning(N)
spectrum = np.fft.fftshift(np.fft.fft(signal * window))
freqs    = np.fft.fftshift(np.fft.fftfreq(N, d=1/OUTPUT_RATE)) / 1e3
mag_db   = 20 * np.log10(np.abs(spectrum) / N + 1e-12)

ax = axes[2]
ax.plot(freqs, mag_db, color="#4CAF50", linewidth=0.6)
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Magnitude (dBFS)")
ax.set_title("Complex FFT Spectrum")
ax.set_ylim(bottom=mag_db.max() - 80)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "./sim_images/cic_output.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")

# ─── Save output as interleaved int16 binary ──────────────────────────────────
rate_khz = int(OUTPUT_RATE / 1e3)
out_path = f"./raw_data_files/decimated_data_{rate_khz}kHz.bin"

interleaved = np.empty(len(signal) * 2, dtype=np.int16)
interleaved[0::2] = signal.real.astype(np.int16)
interleaved[1::2] = signal.imag.astype(np.int16)
interleaved.tofile(out_path)

print(f"\nSaved {len(signal):,} IQ pairs → {out_path}")
print(f"Format : interleaved int16 [I, Q, I, Q, ...]")
print(f"Size   : {os.path.getsize(out_path):,} bytes")

plt.show()