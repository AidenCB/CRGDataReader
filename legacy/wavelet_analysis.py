import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet

# Perform Continuous Wavelet Transform (CWT)
def waveletAnalysis(years, values, x_label="Year", y_label="Data", dj=0.2, w0=6.0, s0_mult=2.0, max_period=24):
    if len(years) < 2 or len(values) < 2:
        raise ValueError("Not enough data points for analysis.")

    dt = np.mean(np.diff(years))
    if np.isnan(dt) or dt <= 0:
        dt = 1 # Fallback if time difference can't be calculated

    mother = wavelet.Morlet(w0)

    # Normalize values to prevent outliers affecting analysis
    std_dev = np.std(values)
    if std_dev > 0:
        normalized_values = (values - np.mean(values)) / std_dev
    else:
        normalized_values = values - np.mean(values)

    # Compute lag-1 autocorrelation for red noise significance test
    if len(normalized_values) < 2:
        raise ValueError("Not enough data points for autocorrelation.")
    alpha_lag1 = np.corrcoef(normalized_values[:-1], normalized_values[1:])[0, 1]

    # Wavelet parameters
    # dj is now passed as a parameter
    s0 = s0_mult * dt  # Smallest scale
    J = int(np.log2(max_period / s0) / dj)

    # Perform CWT
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        normalized_values, dt, dj=dj, s0=s0, J=J, wavelet=mother
    )

    # Compute significance levels
    signif, _ = wavelet.significance(1.0, dt, scales, 0, alpha=alpha_lag1, wavelet=mother)
    power = (np.abs(wave)) ** 2
    sig95 = power / np.outer(signif, np.ones(len(normalized_values)))

    # Plot wavelet power spectrum
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use('dark_background')

    contour = ax.contourf(years, np.log2(scales), power, levels=np.linspace(0, power.max(), 100), cmap='jet')
    fig.colorbar(contour, ax=ax, label='Wavelet Power')
    
    try:
        ax.contour(years, np.log2(scales), sig95, [1], colors='black', linewidths=0.75)
    except Exception:
        pass # Ignores warning if data has no confidence contours to plot
        
    ax.fill_between(years, np.log2(coi), np.log2(scales[-1]), color='gray', alpha=0.3, label='COI')

    ax.set_ylabel('Period (log2 scale)')
    ax.set_xlabel(x_label)
    
    # Dynamically set y-ticks based on max_period
    tick_vals = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    tick_vals = [t for t in tick_vals if t <= max_period]
    if max_period not in tick_vals:
        tick_vals.append(max_period)
    ax.set_yticks(np.log2(tick_vals))
    ax.set_yticklabels([str(int(t)) for t in tick_vals])

    ax.set_title(f'Wavelet Power Spectrum (CWT) of {y_label}')
    ax.set_ylim(np.log2([scales[0], scales[-1]]))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    return fig
