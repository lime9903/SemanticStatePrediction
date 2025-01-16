import numpy as np
from abc import ABC, abstractmethod


class WirelessChannel(ABC):
    """
    Abstract base class for wireless channel environments
    """
    def __init__(self, args):
        self.channel_name = args.channel_name
        self.snr_db = args.snr_db
        self.snr_linear = 10 ** (args.snr_db / 10)

    @abstractmethod
    def apply_channel(self, signal):
        if not isinstance(signal, np.ndarray):
            raise TypeError("Input signal must be a numpy array.")

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
    def __init__(self, args):
        assert args.doppler_freq is not None, "RayleighChannel requires 'doppler_freq' parameter."
        super().__init__(args)
        self.doppler_freq = args.doppler_freq

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
    def __init__(self, args):
        assert args.k_factor is not None
        super().__init__(args)
        self.k_factor = args.k_factor
        self.variance = getattr(args, "variance", 1.0)

    def apply_channel(self, signal):
        k = self.k_factor
        variance = self.variance

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
    def __init__(self, args):
        assert args.m_factor is not None
        super().__init__(args)
        self.m_factor = args.m_factor
        self.omega = args.omega

    def apply_channel(self, signal):
        m = self.m_factor
        omega = self.omega  # Average power

        amplitude = np.sqrt(np.random.gamma(m, omega / m, signal.shape))
        phase = np.random.uniform(0, 2 * np.pi, signal.shape)
        h = amplitude * np.exp(1j * phase)

        faded_signal = h * signal
        faded_signal = faded_signal * np.sqrt(len(signal) / np.sum(np.abs(h) ** 2))

        return self.add_noise(faded_signal)
