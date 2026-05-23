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
from scipy.signal import butter, sosfilt,  firwin, lfilter, resample_poly, hilbert
import matplotlib.pyplot as plt


def upsample_x2_complex(x, h):
    up = np.zeros(len(x) * 2, dtype=np.complex64)
    up[::2] = x
    return lfilter(h, 1.0, up)


# ── 1. Load WAV & print signal info ──────────────────────────────────────────

filename = "./input_files/my_wav_short.wav"
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

# ── 3a. Staged ×2 upsampling using Kaiser halfband FIR ──────────────────────

TARGET_RATE = 1_024_000

def kaiser_halfband():
    # Halfband FIR: efficient for ×2 interpolation
    numtaps = 63  # increase to 127 if you want cleaner stopband
    return firwin(
        numtaps,
        cutoff=0.5,  # normalized (Nyquist = 1 after upsample)
        window=('kaiser', 8.0),
        scale=True
    )

def upsample_x2(x, h):
    up = np.zeros(len(x) * 2, dtype=np.float32)
    up[::2] = x
    return lfilter(h, 1.0, up)

h = kaiser_halfband()

fs = 44100

L_up = L.astype(np.float32)
R_up = R.astype(np.float32)

print(f"  Start Fs: {fs}")

# ×2 stages up to max allowed before exceeding target
while fs * 2 <= TARGET_RATE:
    L_up = upsample_x2(L_up, h)
    R_up = upsample_x2(R_up, h)
    fs *= 2
    print(f"  Upsampled → {fs} Hz")

# If we overshoot or cannot match exactly, go to nearest higher power-of-2 rate
if fs < TARGET_RATE:
    # continue to 1.4112 MHz (next valid ×2 stage)
    while fs * 2 <= 2_000_000:  # safety cap
        L_up = upsample_x2(L_up, h)
        R_up = upsample_x2(R_up, h)
        fs *= 2
        print(f"  Extended upsample → {fs} Hz")

# ── Final rational resample to exactly 1.024 MHz ────────────────────────────

if fs != TARGET_RATE:
    print(f"  Final resample: {fs} → {TARGET_RATE}")

    from math import gcd
    g = gcd(int(fs), TARGET_RATE)

    up = TARGET_RATE // g
    down = int(fs) // g

    L_up = resample_poly(L_up, up, down)
    R_up = resample_poly(R_up, up, down)
    fs = TARGET_RATE

print(f"  Final sample rate: {fs}")

# ── 3. Low-pass filter at 15 kHz ─────────────────────────────────────────────

CUTOFF_HZ  = 15_000          # FM audio bandwidth
LPF_ORDER  = 8               # Butterworth order (steeper → more phase delay)
NYQUIST    = fs / 2

if CUTOFF_HZ >= NYQUIST:
    raise ValueError(
        f"15 kHz cutoff requires sample_rate > 30 kHz; got {sample_rate} Hz"
    )

sos = butter(LPF_ORDER, CUTOFF_HZ / NYQUIST, btype="low", output="sos")

L = L_up
R = R_up
sample_rate = fs

L_filt = sosfilt(sos, L)
R_filt = sosfilt(sos, R)

print(f"  Low-pass filter: {CUTOFF_HZ/1e3:.0f} kHz, Butterworth order {LPF_ORDER}")

# ── 4. Sum & difference ───────────────────────────────────────────────────────

LpR = L_filt + R_filt      # L + R  (mono compatible, 0–15 kHz)
LmR = L_filt - R_filt      # L - R  (stereo difference, modulates sub-carrier)


# ── 6b. Save L+R (LpR) as float32 WAV ────────────────────────────────────────

# Normalise LpR to ±1 for a clean mono export
LpR_norm = LpR / max(np.max(np.abs(LpR)), 1e-9)
lpr_file = "./Intermediate_files/lpr_mono_output.wav"
wav.write(lpr_file, sample_rate, LpR_norm.astype(np.float32))
print(f"  Saved L+R      : {lpr_file}")

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
# MPX = scale * LpR + scale * LmR * subcarr
MPX = scale * LpR + pilot + scale * LmR * subcarr
MPX = MPX.astype(np.float32)

# ── Convert to FM  ────────────────────────────────────

CARRIER_FREQ  = 0e6      # e.g. 105.5 MHz
PEAK_DEVIATION = 75_000     # ±75 kHz (broadcast standard)

# Normalise MPX so its peak maps to exactly ±75 kHz deviation
mpx_peak = np.max(np.abs(MPX)) + 1e-12
kf = PEAK_DEVIATION / mpx_peak   # Hz per unit amplitude

# Running integral of MPX (cumulative sum × dt)
dt = 1.0 / sample_rate
phase = 2 * np.pi * kf * np.cumsum(MPX) * dt

# FM modulated carrier
t = np.arange(len(MPX)) / sample_rate
fm_signal = np.exp(1j * (2 * np.pi * CARRIER_FREQ * t + phase))

# ── Upsample signal ────────────────────────────────────

x = fm_signal
fs = 1_024_000

h = firwin(63, cutoff=0.5, window=('kaiser', 8.0), scale=True)

for i in range(4):
    x = upsample_x2_complex(x, h)
    fs *= 2
    print(f"×2 stage {i+1}: {fs}")

x = resample_poly(x, 3, 1)
fs *= 3

x = resample_poly(x, 5, 1)
fs *= 5

iq = x

# ── Signal level diagnostics ──────────────────────────────────────────────────
real_vals = np.abs(np.real(iq))
imag_vals = np.abs(np.imag(iq))

print(f"  real — max: {np.max(real_vals):.6f}  median: {np.median(real_vals):.6f}")
print(f"  imag — max: {np.max(imag_vals):.6f}  median: {np.median(imag_vals):.6f}")

iq /= np.max(np.abs(iq)) + 1e-12

# ── Signal level diagnostics ──────────────────────────────────────────────────
real_vals = np.abs(np.real(iq))
imag_vals = np.abs(np.imag(iq))

print(f"  real — max: {np.max(real_vals):.6f}  median: {np.median(real_vals):.6f}")
print(f"  imag — max: {np.max(imag_vals):.6f}  median: {np.median(imag_vals):.6f}")

# ── IQ amplitude scale: toggles between -512 and +512 ───────────────────────
IQ_SCALE = 512          # peak int16 value (range -512 … +512)

iq_int16 = np.empty((len(iq) * 2,), dtype=np.int16)
iq_int16[0::2] = (np.real(iq) * IQ_SCALE).astype(np.int16)
iq_int16[1::2] = (np.imag(iq) * IQ_SCALE).astype(np.int16)

iq_int16.tofile("./raw_data_files/iq_245M.bin")
print(f"  Saved: ./raw_data_files/iq_245M.bin  (IQ scale ±{IQ_SCALE})")

# ── Readback verification ─────────────────────────────────────────────────────
verify = np.fromfile("./raw_data_files/iq_245M.bin", dtype=np.int16, count=20)
print("\n--- iq_int16 check ---")

for k in range(10):
    print(f"k = {k} .. iq_int16 value = {iq_int16[k]}")
    print(f"k = {k} .. iq value = {iq[k]}")

print(f"\n  I range: [{verify[0::2].min()}, {verify[0::2].max()}]")
print(f"  Q range: [{verify[1::2].min()}, {verify[1::2].max()}]")

# ── White-noise parameters ────────────────────────────────────────────────────
#   NOISE_INTENSITY : float in [0, 1]
#     0.0  = no noise
#     0.05 = light background noise (−26 dB relative to signal peak)
#     0.5  = heavy noise floor
NOISE_INTENSITY = 0.5      # ← adjust here

rng = np.random.default_rng(seed=42)
noise_i = rng.standard_normal(len(iq)).astype(np.float32) * NOISE_INTENSITY
noise_q = rng.standard_normal(len(iq)).astype(np.float32) * NOISE_INTENSITY

iq_noisy_i = np.real(iq) + noise_i
iq_noisy_q = np.imag(iq) + noise_q

print(f"max(abs(iq_noisy_i)) : {max(abs(iq_noisy_i))}")
print(f"max(abs(iq_noisy_q)) : {max(abs(iq_noisy_q))}")

# Re-normalise so the noisy signal also peaks at ±IQ_SCALE in int16
peak_noisy = np.max(np.sqrt(iq_noisy_i**2 + iq_noisy_q**2)) + 1e-12
iq_noisy_i = iq_noisy_i / peak_noisy
iq_noisy_q = iq_noisy_q / peak_noisy

print(f"max(abs(iq_noisy_i)) (renorm): {max(abs(iq_noisy_i))}")
print(f"max(abs(iq_noisy_q)) (renorm): {max(abs(iq_noisy_q))}")

iq_noisy_int16 = np.empty((len(iq) * 2,), dtype=np.int16)
iq_noisy_int16[0::2] = (iq_noisy_i * IQ_SCALE).astype(np.int16)
iq_noisy_int16[1::2] = (iq_noisy_q * IQ_SCALE).astype(np.int16)

print(f"max(abs(iq_noisy_int16)) (renorm): {max(abs(iq_noisy_int16))}")

iq_noisy_int16.tofile("./raw_data_files/iq_245M_noisy.bin")
print(f"  Saved: ./raw_data_files/iq_245M_noisy.bin  (noise intensity {NOISE_INTENSITY})")



# ── 7. Spectrum plot (MPX baseband) ──────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.suptitle("FM MPX Baseband — Frequency Analysis", fontsize=14, fontweight="bold")

PLOT_SAMPLES = min(N, int(sample_rate * 2))
mpx_freqs = np.fft.rfftfreq(PLOT_SAMPLES, d=1 / sample_rate)

def mag_db(signal):
    spec = np.abs(np.fft.rfft(signal[:PLOT_SAMPLES]))
    spec = np.maximum(spec, 1e-12)
    return 20 * np.log10(spec / np.max(spec))

# ── L+R ──────────────────────────────────────────────────────────────────────
axes[0].plot(mpx_freqs / 1e3, mag_db(LpR), linewidth=0.8)
axes[0].set_title("L+R (mono, 0–15 kHz)")
axes[0].set_xlim(0, 60)
axes[0].set_ylim(-150, 5)
axes[0].set_ylabel("dB")
axes[0].axvline(15, color="red", linestyle="--")
axes[0].grid(True, alpha=0.3)

# ── L−R ──────────────────────────────────────────────────────────────────────
axes[1].plot(mpx_freqs / 1e3, mag_db(LmR), linewidth=0.8)
axes[1].set_title("L−R (stereo difference, 0–15 kHz)")
axes[1].set_xlim(0, 60)
axes[1].set_ylim(-150, 5)
axes[1].set_ylabel("dB")
axes[1].axvline(15, color="red", linestyle="--")
axes[1].grid(True, alpha=0.3)

# ── MPX ───────────────────────────────────────────────────────────────────────
axes[2].plot(mpx_freqs / 1e3, mag_db(MPX), linewidth=0.8)
axes[2].set_title("MPX Baseband (L+R + pilot + L−R DSB-SC)")
axes[2].set_xlim(0, 60)
axes[2].set_ylim(-150, 5)
axes[2].set_xlabel("Frequency (kHz)")
axes[2].set_ylabel("dB")
axes[2].grid(True, alpha=0.3)

# ── Trapezium overlays ────────────────────────────────────────────────────────
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

_trap_top    = -50     # dB: top edge of trapezium
_trap_bot    = -145  # dB: bottom edge of trapezium
_trap_rise   = 1.0   # kHz: ramp width on each edge
_trap_alpha  = 0.13

def _draw_trap(ax, f_lo, f_hi, color, label, label_side="center"):
    """Draw a filled trapezoid between f_lo and f_hi kHz with sloped sides."""
    xs = [f_lo, f_lo + _trap_rise, f_hi - _trap_rise, f_hi,
          f_hi, f_hi - _trap_rise, f_lo + _trap_rise, f_lo]
    ys = [_trap_bot, _trap_top, _trap_top, _trap_bot,
          _trap_bot, _trap_top, _trap_top, _trap_bot]
    # Polygon: top flat, sides sloped
    verts = list(zip(
        [f_lo, f_lo + _trap_rise, f_hi - _trap_rise, f_hi],
        [_trap_bot,         _trap_top, _trap_top, _trap_bot]
    ))
    poly = MplPolygon(verts, closed=True,
                      facecolor=color, edgecolor=color,
                      alpha=_trap_alpha, linewidth=1.2, zorder=2)
    ax.add_patch(poly)
    # Label inside the trapezium
    if label_side == "center":
        x_mid = (f_lo + f_hi) / 2
    elif label_side == "left":
        x_mid = f_lo + (f_hi - f_lo) * 0.3
    else:
        x_mid = f_lo + (f_hi - f_lo) * 0.7
    ax.text(x_mid, -10, label,
            ha="center", va="top", fontsize=8, color=color,
            fontweight="bold", zorder=5)

# L+R  0–15 kHz
_draw_trap(axes[2], 0, 15, color="steelblue",  label="L+R\n0–15 kHz")

# L-R  23–38 kHz  (lower sideband of DSB-SC)
_draw_trap(axes[2], 23, 38, color="darkorange", label="L−R\n23–38 kHz")

# L-R  38–53 kHz  (upper sideband of DSB-SC)
_draw_trap(axes[2], 38, 53, color="darkorange", label="L−R\n38–53 kHz")

# Pilot tone at 19 kHz
axes[2].axvline(19, color="crimson", linestyle=":", linewidth=1.4, zorder=3)
axes[2].text(19, -18, "Pilot\n19 kHz",
             ha="center", va="top", fontsize=8,
             color="crimson", fontweight="bold", zorder=5)

plt.tight_layout()
plt.savefig("./sim_images/fm_mpx_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()

print("  Saved: ./sim_images/fm_mpx_spectrum.png")


# ── Final spectrum at 245.76 MHz (IQ) — clean + noisy ────────────────────────

fs_iq = 245_760_000

PLOT_TIME = 0.002
N_plot = int(PLOT_TIME * fs_iq)

# ── Helper: compute normalised dB spectrum ───────────────────────────────────
def iq_spectrum_db(i_sig, q_sig, n):
    c = (i_sig[:n] + 1j * q_sig[:n]).astype(np.complex64)
    spec = np.fft.fftshift(np.fft.fft(c))
    mag  = 20 * np.log10(np.maximum(np.abs(spec), 1e-12))
    mag -= np.max(mag)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs_iq))
    return freqs, mag

freqs_clean, mag_clean = iq_spectrum_db(np.real(iq),   np.imag(iq),   N_plot)
freqs_noisy, mag_noisy = iq_spectrum_db(iq_noisy_i, iq_noisy_q, N_plot)

fig, (ax_clean, ax_noisy) = plt.subplots(
    1, 2, figsize=(16, 5), sharey=True
)
fig.suptitle("Final IQ Spectrum @ 245.76 MHz", fontsize=13, fontweight="bold")

# Clean
ax_clean.plot(freqs_clean / 1e6, mag_clean, linewidth=0.8, color="steelblue")
ax_clean.set_title("Clean  (no noise)")
ax_clean.set_xlabel("Frequency (MHz)")
ax_clean.set_ylabel("Magnitude (dB)")
ax_clean.set_xlim(-2, 2)
ax_clean.set_ylim(-100, 5)
ax_clean.grid(True, alpha=0.3)
ax_clean.text(
    0.97, 0.97, f"Scale ±{IQ_SCALE}",
    transform=ax_clean.transAxes, ha="right", va="top",
    fontsize=8, color="steelblue"
)

# Noisy
ax_noisy.plot(freqs_noisy / 1e6, mag_noisy, linewidth=0.8, color="darkorange")
ax_noisy.set_title(f"With White Noise  (intensity = {NOISE_INTENSITY})")
ax_noisy.set_xlabel("Frequency (MHz)")
ax_noisy.set_xlim(-2, 2)
ax_noisy.grid(True, alpha=0.3)
ax_noisy.text(
    0.97, 0.97, f"Scale ±{IQ_SCALE}",
    transform=ax_noisy.transAxes, ha="right", va="top",
    fontsize=8, color="darkorange"
)

plt.tight_layout()
plt.savefig("./sim_images/iq_245M_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()

print("  Saved: ./sim_images/iq_245M_spectrum.png")