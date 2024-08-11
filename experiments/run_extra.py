"""DBI methodology simulation — extended analyses.

Adds to run_sim.py:
  1. notch depth vs delay misalignment (theory curve + simulated chain points)
  2. CORDIC NCO accuracy: phase error vs iterations + output-spectrum SFDR
  3. fractional-delay FIR (48-tap Kaiser sinc) magnitude / phase-error response
  4. linear-chirp reconstruction: time-domain error, uncorrected vs calibrated
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dbi_sim import (FS, N, B_DELAY, A_DELAY, AP_R, AP_TH, OV_LO, OV_HI,
                     SPLIT, make_input, acquire, reconstruct, cordic_exp,
                     frac_delay)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

tones = np.arange(0.02, 1.0001, 0.01)
x = make_input(tones)
a, b_if, ref_a, ref_b_if = acquire(x)
from dbi_sim import estimate_delay, tone_amplitudes  # noqa: E402
dA = estimate_delay(ref_a, a)
dB = estimate_delay(ref_b_if, b_if)
d_rel = dB - dA

# ---------------------------------------------------------------- 1. notch vs delay
f_c = 0.55
delta_tau = np.linspace(0, 1.6, 200)          # residual misalignment [samples]
theory = 20 * np.log10(np.abs(0.5 + 0.5 * np.exp(-2j * np.pi * f_c * delta_tau / FS)) + 1e-12)
A_in = tone_amplitudes(x, tones)
ov_sel = (tones >= 0.545) & (tones <= 0.555)

fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.plot(delta_tau, theory, "-", label="theory (crossover centre, wa=wb=0.5)")
y_cal0, _ = reconstruct(a, b_if, d_rel, tones, correct=True)
H_cal0 = tone_amplitudes(y_cal0, tones) / A_in
cal_lvl = 20 * np.log10(H_cal0[ov_sel].mean() / np.mean(H_cal0[(tones > 0.05) & (tones < 0.95)]))
ax.axhline(cal_lvl, color="tab:green", ls="--",
           label=f"calibrated chain operating point ({cal_lvl:.2f} dB, measured)")
ax.set_xlabel("residual delay misalignment [samples]")
ax.set_ylabel("stitched response at crossover [dB]")
ax.set_ylim(-40, 2); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "notch_vs_delay.png"), dpi=160)
sim_pts = [("calibrated chain", cal_lvl)]

# ---------------------------------------------------------------- 2. CORDIC accuracy
ph = np.random.default_rng(3).uniform(-np.pi, np.pi, 2000)
iters_axis = np.arange(1, 25)
max_err = [np.max(np.abs(cordic_exp(ph, iters=i) - np.exp(1j * ph))) for i in iters_axis]
rms_err = [np.sqrt(np.mean(np.abs(cordic_exp(ph, iters=i) - np.exp(1j * ph)) ** 2)) for i in iters_axis]

t = np.arange(N) / FS
nco = cordic_exp(2 * np.pi * SPLIT * t)
spec = np.abs(np.fft.fft(nco * np.exp(-2j * np.pi * SPLIT * t))) / N * 2
car = spec.max()
spurs = np.sort(spec)[::-1]
sfdr = 20 * np.log10(car / spurs[2])   # strongest non-carrier spur

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
ax1.semilogy(iters_axis, max_err, "o-", ms=3, label="max |err|")
ax1.semilogy(iters_axis, rms_err, "s--", ms=3, label="rms |err|")
ax1.set_xlabel("CORDIC iterations"); ax1.set_ylabel("phase error")
ax1.grid(alpha=.3); ax1.legend(fontsize=8)
ax2.plot(20 * np.log10(spec / car + 1e-12)[1:2000], lw=0.6)
ax2.set_xlabel("FFT bin"); ax2.set_ylabel("[dBc]")
ax2.set_title(f"NCO output spectrum, SFDR ≈ {sfdr:.1f} dBc (24 it)", fontsize=9)
ax2.set_ylim(-120, 5); ax2.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "cordic_accuracy.png"), dpi=160)

# ---------------------------------------------------------------- 3. frac-delay FIR response
w = np.linspace(0, np.pi, 800)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
for f_frac, ls in [(0.0, "-"), (0.37, "--"), (0.74, ":")]:
    M = 0
    n = np.arange(48)
    h = np.sinc(n - (24 + f_frac)) * np.kaiser(48, 8.0)
    h /= h.sum()
    H = np.fft.fft(h, 1024)[: 512]
    w_ax = np.linspace(0, np.pi, 512)
    mag_err = 20 * np.log10(np.abs(H) + 1e-12)
    ph_err = np.unwrap(np.angle(H)) + w_ax * (24 + f_frac)
    ax1.plot(w_ax / np.pi, mag_err, ls, lw=1, label=f"frac={f_frac}")
    ax2.plot(w_ax / np.pi, ph_err, ls, lw=1, label=f"frac={f_frac}")
ax1.set_ylim(-1.5, 0.5); ax1.set_xlabel("w / pi"); ax1.set_ylabel("|H| [dB]")
ax1.set_title("fractional-delay FIR magnitude", fontsize=9)
ax2.set_xlabel("w / pi"); ax2.set_ylabel("phase [rad]")
ax2.set_title("residual phase after ideal-delay removal", fontsize=9)
for ax in (ax1, ax2):
    ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "frac_delay_response.png"), dpi=160)

# ---------------------------------------------------------------- 4. chirp reconstruction
T = N / FS
tt = np.arange(N) / FS
f0, f1 = 0.05, 0.95
xc = np.exp(1j * 2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * T)))
a_c, b_c, ra_c, rb_c = None, None, None, None
from dbi_sim import acquire as _acquire  # noqa: E402
a_c, b_c, ra_c, rb_c = _acquire(xc)
y_raw_c, _ = reconstruct(a_c, b_c, d_rel, tones, correct=False)
y_cal_c, _ = reconstruct(a_c, b_c, d_rel, tones, correct=True)
def align_err(y, xref):
    d = estimate_delay(xref, y)
    ya = frac_delay(y, -d)
    err = np.abs(ya - xref)
    k = 101
    env = np.convolve(err, np.ones(k) / k, mode="same")
    return env
env_raw = align_err(y_raw_c, xc)
env_cal = align_err(y_cal_c, xc)

fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.plot(tt, env_raw, lw=1, label="uncorrected reconstruction error")
ax.plot(tt, env_cal, lw=1, label="calibrated reconstruction error")
ax.axvspan(OV_LO * T / 1.0, OV_HI * T / 1.0, alpha=0.15, color="tab:red")
ax.set_xlabel("time [samples/FS]"); ax.set_ylabel("|error| (smoothed)")
ax.set_title("linear-chirp reconstruction (0.05-0.95 sweep)", fontsize=9)
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "chirp_recon.png"), dpi=160)

print("EXTRA_DONE")
print(f"calibrated chain crossover level: {cal_lvl:.2f} dB")
print(f"CORDIC 24-it max err: {max_err[-1]:.2e} | NCO SFDR: {sfdr:.1f} dBc")
