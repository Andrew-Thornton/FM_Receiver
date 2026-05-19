"""
FM MPX (Multiplex) Signal Constructor
======================================
Steps:
  1. Load WAV file and detect sample rate / bit depth
  2. Extract L and R channels
  3. Low-pass filter both channels at 15 kHz
  4. Form sum  (L+R) and difference (L-R) signals
  5. Construct the MPX baseband:
       MPX(t) = (L+R)  +  pilot·sin(2π·19kHz·t)  +  (L-R)·cos(2π·38kHz·t)
"""

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt

# ── 1. Load WAV & print signal info ──────────────────────────────────────────

filename = "my_wav.wav"
sample_rate, data = wav.read(filename)

print("=" * 50)
print(f"  File          : {filename}")
print(f"  Sample rate   : {sample_rate} Hz")
print(f"  Bit depth     : {data.dtype}")
print(f"  Channels      : {data.ndim if data.ndim == 1 else data.shape[1]}")
print(f"  Total samples : {data.shape[0]}")
print(f"  Duration      : {data.shape[0] / sample_rate:.3f} s")
print("=" * 50)

# ── 2. Extract L / R channels ─────────────────────────────────────────────────

# Normalise to float32 in [-1, 1]
if data.dtype == np.int16:
    audio = data.astype(np.float32) / 32768.0
elif data.dtype == np.int32:
    audio = data.astype(np.float32) / 2147483648.0
elif data.dtype == np.uint8:
    audio = (data.astype(np.float32) - 128.0) / 128.0
else:
    audio = data.astype(np.float32)

if audio.ndim == 1:
    # Mono input → treat as L=R
    print("  [!] Mono file detected — duplicating to L and R")
    L = audio.copy()
    R = audio.copy()
else:
    L = audio[:, 0]
    R = audio[:, 1]

print(f"  L peak: {np.max(np.abs(L)):.4f}   R peak: {np.max(np.abs(R)):.4f}")

# ── 3. Low-pass filter at 15 kHz ─────────────────────────────────────────────

CUTOFF_HZ  = 15_000          # FM audio bandwidth
LPF_ORDER  = 8               # Butterworth order (steeper → more phase delay)
NYQUIST    = sample_rate / 2

if CUTOFF_HZ >= NYQUIST:
    raise ValueError(
        f"15 kHz cutoff requires sample_rate > 30 kHz; got {sample_rate} Hz"
    )

sos = butter(LPF_ORDER, CUTOFF_HZ / NYQUIST, btype="low", output="sos")
L_filt = sosfilt(sos, L)
R_filt = sosfilt(sos, R)

print(f"  Low-pass filter: {CUTOFF_HZ/1e3:.0f} kHz, Butterworth order {LPF_ORDER}")

# ── 4. Sum & difference ───────────────────────────────────────────────────────

LpR = L_filt + R_filt      # L + R  (mono compatible, 0–15 kHz)
LmR = L_filt - R_filt      # L - R  (stereo difference, modulates sub-carrier)

# ── 5. Build MPX baseband signal ──────────────────────────────────────────────
#
#   MPX(t) = (L+R)
#           + 0.1 · sin(2π · 19 kHz · t)          ← pilot tone
#           + (L-R) · cos(2π · 38 kHz · t)         ← DSB-SC sub-carrier
#
# Amplitude conventions (IEC 62106 / typical broadcast):
#   (L+R)  → ±45 % modulation
#   pilot  → 10 % modulation
#   (L-R)  → ±45 % modulation
#
PILOT_FREQ   = 19_000        # Hz
SUBCARR_FREQ = 38_000        # Hz  (= 2 × pilot)
PILOT_AMP    = 0.1           # 10 %

N = len(LpR)
t = np.arange(N) / sample_rate

pilot    = PILOT_AMP * np.sin(2 * np.pi * PILOT_FREQ   * t)
subcarr  =             np.cos(2 * np.pi * SUBCARR_FREQ * t)

# Scale L+R and L-R to ±0.45 so headroom remains for pilot
scale = 0.45 / max(np.max(np.abs(LpR)), np.max(np.abs(LmR)), 1e-9)
MPX = scale * LpR + pilot + scale * LmR * subcarr

# Normalise MPX to ±1 for output
MPX /= np.max(np.abs(MPX))

print(f"  MPX samples    : {len(MPX)}")
print(f"  MPX peak       : {np.max(np.abs(MPX)):.4f}")

# ── 6. Save MPX as float32 WAV ────────────────────────────────────────────────

out_file = "fm_mpx_output.wav"
wav.write(out_file, sample_rate, MPX.astype(np.float32))
print(f"  Saved          : {out_file}")

# ── 7. Spectrum plot ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.suptitle("FM MPX Baseband — Frequency Analysis", fontsize=14, fontweight="bold")

PLOT_SAMPLES = min(N, sample_rate * 2)   # plot up to 2 s to keep FFT fast
freqs = np.fft.rfftfreq(PLOT_SAMPLES, d=1 / sample_rate)

def mag_db(signal):
    spec = np.abs(np.fft.rfft(signal[:PLOT_SAMPLES]))
    spec = np.maximum(spec, 1e-12)
    return 20 * np.log10(spec / np.max(spec))

# L+R spectrum
axes[0].plot(freqs / 1e3, mag_db(LpR), color="#2196F3", linewidth=0.8)
axes[0].set_title("L+R  (mono, 0–15 kHz)")
axes[0].set_xlim(0, 60)
axes[0].set_ylim(-80, 5)
axes[0].set_ylabel("dB")
axes[0].axvline(15, color="red", linestyle="--", linewidth=0.8, label="15 kHz LPF")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# L-R spectrum
axes[1].plot(freqs / 1e3, mag_db(LmR), color="#4CAF50", linewidth=0.8)
axes[1].set_title("L−R  (stereo difference, 0–15 kHz)")
axes[1].set_xlim(0, 60)
axes[1].set_ylim(-80, 5)
axes[1].set_ylabel("dB")
axes[1].axvline(15, color="red", linestyle="--", linewidth=0.8, label="15 kHz LPF")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Full MPX spectrum
axes[2].plot(freqs / 1e3, mag_db(MPX), color="#FF5722", linewidth=0.8)
axes[2].set_title("MPX Baseband  (L+R) + pilot@19kHz + (L−R)·cos(38kHz)")
axes[2].set_xlim(0, 60)
axes[2].set_ylim(-80, 5)
axes[2].set_xlabel("Frequency (kHz)")
axes[2].set_ylabel("dB")
for freq, label in [(15, "15 kHz"), (19, "Pilot\n19 kHz"), (23, ""), (38, "Sub-carrier\n38 kHz"), (53, "")]:
    axes[2].axvline(freq, color="gray", linestyle=":", linewidth=0.8)
axes[2].annotate("Pilot 19 kHz",  xy=(19, -5), fontsize=7, color="gray", ha="center")
axes[2].annotate("38 kHz SC",     xy=(38, -5), fontsize=7, color="gray", ha="center")
axes[2].annotate("23–53 kHz DSB-SC", xy=(38, -12), fontsize=7, color="gray", ha="center")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fm_mpx_spectrum.png", dpi=150, bbox_inches="tight")
print("  Spectrum plot  : fm_mpx_spectrum.png")
plt.show()