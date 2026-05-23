import numpy as np
import matplotlib.pyplot as plt
import os
import sys

FILE_PATH  = "./raw_data_files/decimated_data_120kHz.bin"
INPUT_RATE = 120e3  # Hz

# ─── Load IQ file ─────────────────────────────────────────────────────────────
if not os.path.exists(FILE_PATH):
    print(f"ERROR: File not found: {FILE_PATH}")
    sys.exit(1)

file_bytes = os.path.getsize(FILE_PATH)
raw = np.fromfile(FILE_PATH, dtype=np.int16, count=(file_bytes // 2))
iq  = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
print(f"Loaded {len(iq):,} IQ samples @ {INPUT_RATE/1e3:.1f} kHz")

t = np.arange(len(iq)) / INPUT_RATE * 1e3  # ms

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Instantaneous phase via arctan2 + unwrap
# ═══════════════════════════════════════════════════════════════════════════════
#
# np.angle() computes atan2(Q, I), giving the wrapped phase in (-π, π].
# np.unwrap() then removes the 2π discontinuities so the phase grows
# continuously, making it easy to differentiate into instantaneous frequency.

phase_wrapped   = np.angle(iq)                   # radians, wrapped to (-π, π]
phase_unwrapped = np.unwrap(phase_wrapped)        # radians, continuous

# Instantaneous frequency = d(phase)/dt  (in Hz)
inst_freq_arctan = np.diff(phase_unwrapped) / (2 * np.pi / INPUT_RATE)

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Phase via a Phase-Locked Loop (PLL)
# ═══════════════════════════════════════════════════════════════════════════════
#
# A 2nd-order PLL tracks the carrier by driving a numerically-controlled
# oscillator (NCO) to follow the input phase.  The loop filter (PI controller)
# feeds two accumulators:
#
#   phase_error  = Im{ input * conj(NCO) }   (cross-product discriminant)
#   integrator  += Kp * error                (removes steady-state freq offset)
#   freq        += Ki * error                (fine frequency correction)
#   nco_phase   += freq                      (NCO advances by current estimate)
#
# Tune bandwidth with loop_bw (normalised, 0 < loop_bw < 0.5).
# Wider → faster lock, more noise.  Narrower → slower, cleaner.

def run_pll(iq_signal, loop_bw=0.01, damping=0.707):
    """
    2nd-order PLL.  Returns the NCO phase array (radians, unwrapped).

    Parameters
    ----------
    iq_signal : complex array
    loop_bw   : normalised loop bandwidth  (fraction of sample rate, e.g. 0.01)
    damping   : damping factor (0.707 = critically damped / Butterworth)
    """
    n = len(iq_signal)

    # Compute PI gains from loop bandwidth and damping factor
    # (Gardner / Proakis design equations)
    Bn   = loop_bw                              # normalised noise bandwidth
    K0   = 1.0                                  # NCO gain (normalised)
    Kd   = 1.0                                  # discriminator gain
    theta_n = Bn / (damping + 1 / (4 * damping))
    Kp   = (4 * damping * theta_n) / (1 + 2 * damping * theta_n + theta_n**2) / (K0 * Kd)
    Ki   = (4 * theta_n**2)        / (1 + 2 * damping * theta_n + theta_n**2) / (K0 * Kd)

    print(f"PLL  loop_bw={loop_bw}  Kp={Kp:.5f}  Ki={Ki:.6f}")

    nco_phase    = np.zeros(n)
    integrator   = 0.0
    freq         = 0.0          # current frequency estimate (rad/sample)
    current_phase = 0.0

    for i in range(n):
        # NCO output
        nco    = np.exp(1j * current_phase)

        # Phase error: imaginary part of input * conj(NCO)  (small-angle approx)
        error  = np.imag(iq_signal[i] * np.conj(nco))

        # PI loop filter
        integrator   += Ki * error
        freq          = Kp * error + integrator

        # Advance NCO
        current_phase += freq
        nco_phase[i]   = current_phase

    return nco_phase, Kp, Ki

phase_pll, Kp, Ki = run_pll(iq, loop_bw=0.01)

# Instantaneous frequency from PLL phase (rad/sample → Hz)
inst_freq_pll = np.diff(phase_pll) / (2 * np.pi / INPUT_RATE)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 14))
fig.suptitle("Phase Extraction — arctan+unwrap  vs  PLL", fontsize=13, fontweight="bold")

# ── 1. Wrapped phase ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(t, np.degrees(phase_wrapped), color="#9C27B0", linewidth=0.5)
ax.set_ylabel("Phase (°)")
ax.set_title("Wrapped Phase  [atan2(Q, I)]")
ax.set_ylim(-185, 185)
ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
ax.grid(True, alpha=0.3)

# ── 2. Unwrapped phase (both methods) ─────────────────────────────────────────
ax = axes[1]
ax.plot(t, np.degrees(phase_unwrapped), color="#2196F3", linewidth=0.7, label="arctan + unwrap")
ax.plot(t, np.degrees(phase_pll),       color="#F44336", linewidth=0.7, alpha=0.8, label="PLL")
ax.set_ylabel("Phase (°)")
ax.set_title("Unwrapped Phase Comparison")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# ── 3. Instantaneous frequency — arctan method ───────────────────────────────
ax = axes[2]
ax.plot(t[1:], inst_freq_arctan, color="#2196F3", linewidth=0.5)
ax.set_ylabel("Freq (Hz)")
ax.set_title("Instantaneous Frequency — arctan + unwrap  (d φ/dt)")
ax.grid(True, alpha=0.3)

# ── 4. Instantaneous frequency — PLL ─────────────────────────────────────────
ax = axes[3]
ax.plot(t[1:], inst_freq_pll, color="#F44336", linewidth=0.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Freq (Hz)")
ax.set_title(f"Instantaneous Frequency — PLL  (loop_bw=0.01, Kp={Kp:.4f}, Ki={Ki:.5f})")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("phase_extraction.png", dpi=150)
plt.show()
print("Saved: phase_extraction.png")