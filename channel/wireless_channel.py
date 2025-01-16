import numpy as np
from abc import ABC, abstractmethod


class WirelessChannel(ABC):
    """
    Abstract base class for wireless channel environments
    """
    def __init__(self, channel_name, snr_db):
        self.channel_name = channel_name
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)

    @abstractmethod
    def apply_channel(self, signal):
        pass

    def get_snr(self, db=True):
        if db: return self.snr_db
        else: return self.snr_linear

    def add_noise(self, signal):
        """
        Add AWGN noise to the signal based on SNR
        """
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / self.snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, signal.shape) +
                                          1j * np.random.normal(0, 1, signal.shape))
        return signal + noise


class AWGNChannel(WirelessChannel):
    """
    Additive White Gaussian Noise (AWGN) channel
    """
    def apply_channel(self, signal):
        return self.add_noise(signal)


class RayleighChannel(WirelessChannel):
    """
    Rayleigh fading channel with AWGN
    """
    def __init__(self, channel_name, snr_db, doppler_freq=0):
        super().__init__(channel_name, snr_db)
        self.doppler_freq = doppler_freq

    def apply_channel(self, signal):
        # Generate Rayleigh fading coefficients
        h = np.sqrt(1/2) * (np.random.normal(0, 1, signal.shape) +
                            1j * np.random.normal(0, 1, signal.shape))

        faded_signal = h * signal
        faded_signal = faded_signal * np.sqrt(len(signal) / np.sum(np.abs(h) ** 2))

        return self.add_noise(faded_signal)


class RicianChannel(WirelessChannel):
    """
    Rician fading channel with AWGN
    """
    def __init__(self, channel_name, snr_db, k_factor):
        super().__init__(channel_name, snr_db)
        self.k_factor = k_factor

    def apply_channel(self, signal):
        k = self.k_factor
        variance = 1

        # LOS (specular) component
        los = np.sqrt(k * variance / (k + 1))

        # NLOS (scattered) component
        nlos = np.sqrt(variance / (2 * (k + 1))) * (np.random.normal(0, 1, signal.shape) +
                                                   1j * np.random.normal(0, 1, signal.shape))

        h = los + nlos

        faded_signal = h * signal
        faded_signal = faded_signal * np.sqrt(len(signal) / np.sum(np.abs(h) ** 2))

        return self.add_noise(faded_signal)


class NakagamiChannel(WirelessChannel):
    """
    Nakagami fading channel with AWGN
    """
    def __init__(self, channel_name, snr_db, m_factor):
        super().__init__(channel_name, snr_db)
        self.m_factor = m_factor

    def apply_channel(self, signal):
        m = self.m_factor
        omega = 1  # Average power

        amplitude = np.sqrt(np.random.gamma(m, omega/m, signal.shape))
        phase = np.random.normal(0, 2*np.pi, signal.shape)
        h = amplitude * np.exp(1j * phase)

        faded_signal = h * signal
        faded_signal = faded_signal * np.sqrt(len(signal) / np.sum(np.abs(h) ** 2))

        return self.add_noise(faded_signal)

