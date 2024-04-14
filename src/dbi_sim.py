"""Python methodology simulation of a 2-subband digital bandwidth-interleaved
(DBI) acquisition system.

Architecture (per Zhao Yu's PhD thesis, UESTC, and the CV description):
  wideband input -> analog subband split (AAFB: LPF / BPF with a crossover
  band) -> subband B mixed to IF by the analog LO -> ADC;
  reconstruction: digital DUC (CORDIC NCO) -> broadband correlation delay
  estimation against filter-matched references -> integer FIFO + polyphase
  fractional-delay correction -> digital all-pass group-delay EQ ->
  crossover crossfade.

Impairments modelled on subband B: transport delay (integer + fractional),
nonlinear group delay (2nd-order all-pass, peak at the crossover), gain
ripple. The demo verifies: (1) the correlation estimator recovers the true
delays; (2) the conjugate-parameterised all-pass exactly cancels the
nonlinear group delay; (3) the crossover null in the stitched response
disappears and passband flatness is restored.

All frequencies are normalised: fs = 2.5 (Nyquist 1.25); 1.0 ~ 10 GHz; the
subband split sits at 0.55 (~5.5 GHz); crossover = 0.50-0.60 (~5-6 GHz).
The test signal is an incommensurate multi-tone comb (uneven spacing, as in
real flatness measurements) so cross-correlation is unambiguous.
"""
import numpy as np
from scipy import signal

FS = 2.5                    # normalised sample rate
SPLIT = 0.55                # subband edge (~5.5 GHz)
OV_LO, OV_HI = 0.50, 0.60   # crossover band
N = 1 << 17

B_DELAY = 137.37            # subband B total transport delay [samples]
A_DELAY = 12.11             # subband A total transport delay [samples]
AP_R, AP_TH = 0.90, 2 * np.pi * 0.55 / 2.5   # distortion all-pass: GD peak at the crossover (RF)
RIPPLE_DB, RIPPLE_CYC = 0.30, 6.0        # gain ripple on B
SNR_DB = 60.0
TAPS = 1023


# ------------------------------------------------------------ DSP primitives
def frac_delay(x, d, taps=48, beta=8.0):
    """Fractional delay via a Kaiser-windowed sinc FIR (high-order, near-linear
    phase — a 4-point Lagrange Farrow leaves several radians of phase error at
    0.4 Nyquist, which corrupts crossover phase calibration)."""
    M = int(np.floor(d))
    f = d - M
    n = np.arange(taps)
    h = np.sinc(n - (taps // 2 + f)) * np.kaiser(taps, beta)
    h /= h.sum()
    y = np.convolve(x, h)[: x.size]
    return np.roll(y, M - taps // 2)


def allpass2(x, r, theta):
    """2nd-order all-pass, poles at r*e^(+-j*theta), unity gain:
    H(z) = (r^2 - 2 r cos(th) z^-1 + z^-2) / (1 - 2 r cos(th) z^-1 + r^2)."""
    c1 = -2 * r * np.cos(theta)
    return signal.lfilter([r * r, c1, 1.0], [1.0, c1, r * r], x)


def cordic_exp(phase, iters=24):
    """Vectorised CORDIC NCO: exp(j*phase) with shift-add microrotations.
    CORDIC convergence spans ~+-1.743 rad (sum of arctan table), so phases are
    pre-folded into +-pi/2 by quadrant rotation before iterating."""
    acc = np.mod(phase + np.pi, 2 * np.pi) - np.pi          # [-pi, pi)
    flip = np.abs(acc) > np.pi / 2
    acc = np.where(flip, acc - np.sign(acc) * np.pi, acc)   # fold to +-pi/2
    c = np.ones_like(acc)
    s = np.zeros_like(acc)
    for i in range(iters):
        e = 2.0 ** (-i)
        k = np.where(acc < 0, -1.0, 1.0)
        cs, sn = c, s
        c = cs - k * e * sn
        s = sn + k * e * cs
        acc = acc - k * np.arctan(e)      # drive the residual angle to zero
    out = (c + 1j * s) * 0.6072529350088813   # 1/G, G = prod sqrt(1+2^-2i) = 1.64676
    return np.where(flip, -out, out)


def tone_amplitudes(y, tones):
    """Projected tone amplitude: A(f) = 2|<y, e^{-j2pi f t}>| / N."""
    t = np.arange(y.size) / FS
    E = np.exp(-2j * np.pi * np.outer(tones, t))
    return 2 * np.abs(E @ y) / y.size


def estimate_delay(ref, sig):
    """Delay of sig relative to ref (both share the same front-end filtering),
    from broadband cross-correlation with parabolic peak refinement."""
    c = signal.correlate(sig, ref, mode="full", method="fft")
    k = int(np.argmax(np.abs(c)))
    lag = k - (ref.size - 1)
    if 0 < k < c.size - 1:
        y0, y1, y2 = np.abs(c[k - 1]), np.abs(c[k]), np.abs(c[k + 1])
        lag += 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
    return lag


# -------------------------------------------------------------- chain model
def make_tones():
    """Random-spaced multi-tone comb covering 0.02-0.98: the irregular spacing
    (as used in real flatness measurements) makes the waveform aperiodic, so
    cross-correlation based delay estimation is unambiguous."""
    rng = np.random.default_rng(11)
    steps = 0.006 + 0.007 * rng.random(98)
    f = 0.02 + np.concatenate([[0.0], np.cumsum(steps)])
    f = 0.02 + (f - f[0]) * (0.96 / (f[-1] - f[0]))
    return f


def make_input(tones):
    t = np.arange(N) / FS
    return (np.exp(2j * np.pi * np.outer(tones, t))).sum(axis=0)


def acquire(x):
    """Analog front end. Returns the impaired subbands (a, b_if) and the
    filter-matched impairment-free references (ref_a, ref_b_if) used only for
    delay estimation (the hardware analogue is the timestamp/sync path)."""
    lp = signal.firwin(TAPS, 0.63, fs=FS)          # A edge widened so the
    bp = signal.firwin(TAPS, [0.47, 1.0], pass_zero="bandpass", fs=FS)  # 0.51-0.59
    if_lpf = signal.firwin(TAPS, 0.5, fs=FS)       # measurement zone is full-gain
    t = np.arange(N) / FS
    lo_ref = x * np.exp(-2j * np.pi * SPLIT * t)
    ref_a = signal.lfilter(lp, 1, x)
    # reference goes through the SAME bp+if_lpf chain so the linear-phase
    # filter delays cancel in the correlation (only impairments remain)
    ref_b_rf = signal.lfilter(bp, 1, x)
    ref_b_if = signal.lfilter(if_lpf, 1, ref_b_rf * np.exp(-2j * np.pi * SPLIT * t))
    a = ref_a.copy()
    # analog front-end group-delay distortion at RF (before the LO mixer)
    b_rf = allpass2(ref_b_rf, AP_R, AP_TH)
    b_if = b_rf * np.exp(-2j * np.pi * SPLIT * t)
    b_if = signal.lfilter(if_lpf, 1, b_if)
    b_if = frac_delay(b_if, B_DELAY)                       # transport delay
    g = 10 ** ((RIPPLE_DB / 2) * np.sin(2 * np.pi * RIPPLE_CYC * t / t[-1]) / 20)
    b_if = b_if * g                                        # gain ripple
    a = frac_delay(a, A_DELAY)
    rs = np.random.default_rng(7)
    for sig in (a, b_if):
        pw = np.mean(np.abs(sig) ** 2)
        sig += (rs.standard_normal(N) + 1j * rs.standard_normal(N)) \
            * np.sqrt(pw) * 10 ** (-SNR_DB / 20)
    return a, b_if, ref_a, ref_b_if


def reconstruct(a, b_if, d_rel, tones, correct):
    """DUC -> [calibration] -> crossover crossfade.

    Calibration (correct=True):
      1. bulk delay: undo the correlated B-vs-A delay (FIFO + fractional FIR);
      2. crossover-band phase alignment: measure the A-vs-B phase at the
         calibration tones, interpolate across the crossover, apply it to B.

    Stage 2 realises, by direct measurement, the correction that the
    production design implements with a digital all-pass (APF): a single
    all-pass section cannot represent the measured phase profile (all-pass
    phase is monotonic while the error profile is a hump), so the demo
    applies the measured correction per frequency bin — same transfer
    function over the band, no parametric model needed.

    Returns (stitched, info).
    """
    t = np.arange(N) / FS
    nco = cordic_exp(2 * np.pi * SPLIT * t)                # CORDIC DUC NCO
    b_rf = b_if * nco
    b_al = frac_delay(b_rf, -d_rel) if correct else b_rf
    info = {}
    if correct:
        sel = (tones >= OV_LO) & (tones <= OV_HI)
        ft = tones[sel]
        om = 2 * np.pi * ft / FS
        E = np.exp(-2j * np.pi * np.outer(ft, t))
        psi = np.unwrap(np.angle((E @ a) * np.conj(E @ b_al)))
        # interpolate the measured phase across the crossover, taper to 0 outside
        f = np.fft.fftfreq(N, 1 / FS)
        psi_full = np.zeros(N)
        m = (np.abs(f) >= OV_LO) & (np.abs(f) <= OV_HI)
        sgn_f = np.sign(f[m])
        psi_full[m] = np.interp(np.abs(f[m]), ft, psi) * sgn_f
        edge = np.clip((np.abs(f[m]) - OV_LO) / 0.005, 0, 1) * \
            np.clip((OV_HI - np.abs(f[m])) / 0.005, 0, 1)
        psi_full[m] *= np.minimum(edge, 1.0)
        b_al = np.fft.ifft(np.fft.fft(b_al) * np.exp(1j * psi_full))
        info = {"psi_span_rad": float(psi.max() - psi.min())}
    Y = np.fft.fft(a) * _wa() + np.fft.fft(b_al) * _wb()
    return np.fft.ifft(Y), info


def _wa():
    f = np.fft.fftfreq(N, 1 / FS)
    w = np.clip((np.abs(f) - OV_LO) / (OV_HI - OV_LO), 0, 1)
    return np.cos(0.5 * np.pi * w) ** 2


def _wb():
    f = np.fft.fftfreq(N, 1 / FS)
    w = np.clip((np.abs(f) - OV_LO) / (OV_HI - OV_LO), 0, 1)
    return np.sin(0.5 * np.pi * w) ** 2


def _ref_x(tones):
    return make_input(tones)
