# DBI methodology simulation — summary (this run, synthetic signals)

- True delays: A 12.11 samples, B 137.37 samples; correlation estimates: A 12.107, B 139.932; B-vs-A true 125.26 vs estimated 127.825 (error +2.565 samples — the residual is the front-end all-pass group-delay spread, removed by the phase alignment)
- Front-end distortion all-pass: r=0.9, theta=0.44*pi (group-delay peak ~19 samples at the crossover)
- Crossover phase alignment: measured correction span 8.53 rad across the band
- Crossover-band stitched response: uncorrected -7.1..+0.0 dB (null depth -7.1 dB); corrected -0.03..+0.02 dB
- Overall passband ripple (0.05-0.95): uncorrected 7.17 dB -> corrected 0.06 dB
- Residuals by design: gain ripple 0.30 dB p-p on B (modelled, not equalised)
- Reference: the thesis reports its crossover compensation structure reduces the overlap-band amplitude impact to 0.33 dB (chapter 1); this demo is a methodology verification, not a reproduction of the thesis hardware numbers.
