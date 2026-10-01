import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet
from pycwt.helpers import find


# Perform Continuous Wavelet Transform (CWT)
def waveletAnalysis(years, values):
    dt = 1  # Time step in years
    mother = wavelet.Morlet(6)

    # Normalize values to prevent outliers affecting analysis
    values = (values - np.mean(values)) / np.std(values)

    # Compute lag-1 autocorrelation for red noise significance test
    alpha_lag1 = np.corrcoef(values[:-1], values[1:])[0, 1]

    # Wavelet parameters
    dj = 0.2  # Scale resolution (smaller = finer detail)
    s0 = 2 * dt  # Smallest scale (2-year periods)
    maxPeriod = 24  # Largest period to analyze
    J = int(np.log2(maxPeriod / s0) / dj)  # Number of scales

    # Perform CWT
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        values, dt, dj=dj, s0=s0, J=J, wavelet=mother
    )

    # Compute significance levels using lag-1 autocorrelation
    signif, fft_theor = wavelet.significance(
        np.std(values) ** 2, dt, scales, 0, alpha=alpha_lag1, wavelet=mother
    )
    sig95 = np.abs(wave) ** 2 / signif[:, None]

    # Normalize power for visualization
    power = np.abs(wave) ** 2
    power /= np.max(power)

    # Plot wavelet power spectrum
    plt.figure(figsize=(12, 6))
    plt.contourf(
        years, np.log2(scales), power,
        levels=np.linspace(0, np.max(power), 100), cmap='jet'
    )
    plt.colorbar(label='Wavelet Power')

    # Significance contour (95% confidence)
    plt.contour(years, np.log2(scales), sig95 > 1, levels=[0], colors='black', linewidths=0.75)

    # Cone of Influence (COI)
    plt.fill_between(years, np.log2(coi), np.log2(scales[-1]), color='gray', alpha=0.3, label='COI')

    # Labels and formatting
    plt.ylabel('Period (Years, log2 scale)')
    plt.xlabel('Year')
    plt.title('Wavelet Power Spectrum (CWT) of Newark Annual Precipitation')
    plt.yticks(np.log2([2, 4, 8, 10, 16, 20, 24]), labels=['2', '4', '8', '10', '16', '20', '24'])
    plt.ylim(np.log2([scales[0], scales[-1]]))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


# Load the data
def load_data(filename):
    years = []
    values = []

    with open(filename, 'r') as file:
        for line in file:
            # 1. Handle different delimiters by converting commas to spaces, then strip whitespace
            line = line.replace(',', ' ').strip()

            # 2. Skip completely empty rows or comment lines
            if not line or line.startswith(('#', '%')):
                continue

            # Split by any amount of whitespace
            parts = line.split()
            
            if len(parts) >= 2:
                try:
                    # 3. Attempt to cast the first two columns to floats
                    y = float(parts[0])
                    v = float(parts[1])
                    years.append(y)
                    values.append(v)
                except ValueError:
                    # 4. If casting fails (e.g., it hits text like "Year", "Value"), skip the row
                    continue

    years = np.array(years)
    values = np.array(values)

    return years, values


# Line graph to display raw data
def rawGraph(years, values):
    plt.figure(figsize=(10, 5))
    plt.plot(years, values)
    plt.xlabel("Year")
    plt.ylabel("Annual Precipitation (inches)")
    plt.title("Raw Time Series Data — Annual Precipitation")
    plt.grid()
    plt.show()


# Main execution
if __name__ == "__main__":
    filename = "data/CRGData/NewarkLaGuardiaPrecip.txt"
    years, values = load_data(filename)
    # rawGraph(years, values)
    waveletAnalysis(years, values)
