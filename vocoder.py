"""Streaming filter-bank vocoder with a built-in, band-limited synth carrier."""
import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt, lfilter

VOCODER_DEFAULTS = dict(vocoder_enabled=False, vocoder_mix=.85, vocoder_bands=24,
                        vocoder_carrier='saw', vocoder_note=45, vocoder_chord='unison',
                        vocoder_detune=6, vocoder_pulse_width=.5, vocoder_noise=.08,
                        vocoder_attack_ms=5, vocoder_release_ms=70,
                        vocoder_formant=0, vocoder_bandwidth=1, vocoder_consonants=.3,
                        vocoder_gain_db=0, vocoder_freeze=False)
CHORDS = {'unison': (0,), 'octave': (0, 12), 'fifth': (0, 7),
          'major': (0, 4, 7), 'minor': (0, 3, 7)}
CARRIERS = ('saw', 'square', 'pulse', 'noise')


@njit(cache=True)
def follow_envelope(values, previous, attack, release):
    """Per-sample attack/release, compiled to keep 32 bands out of Python loops."""
    result = np.empty(len(values), dtype=np.float64)
    for i in range(len(values)):
        target = abs(values[i])
        coefficient = attack if target > previous else release
        previous = coefficient * previous + (1 - coefficient) * target
        result[i] = previous
    return result, previous


def poly_blep(phase, step):
    # Correct waveform discontinuities to reduce oscillator aliasing.
    result = np.zeros_like(phase)
    first = phase < step
    t = phase[first] / step
    result[first] = 2 * t - t * t - 1
    last = phase > 1 - step
    t = (phase[last] - 1) / step
    result[last] = t * t + 2 * t + 1
    return result


class Vocoder:
    def __init__(self, sample_rate=44100, seed=None):
        self.sr = sample_rate
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.bank_key = None
        self.phases = np.zeros(6)
        self.frozen = None

    def configure(self, bands, width):
        self.centers = np.geomspace(100, min(8000, self.sr * .4), bands)
        spacing = self.centers[1] / self.centers[0]
        self.filters = []
        for center in self.centers:
            low = max(40, center / spacing ** (width / 2))
            high = min(self.sr * .46, center * spacing ** (width / 2))
            self.filters.append(butter(2, [low, high], btype='bandpass', fs=self.sr, output='sos'))
        self.analysis_state = [np.zeros((2, 2)) for _ in self.filters]
        self.synthesis_state = [np.zeros((2, 2)) for _ in self.filters]
        self.envelopes = np.zeros(bands)
        self.carrier_power = np.zeros((bands, 1))
        self.consonant_filter = butter(2, 3500, btype='highpass', fs=self.sr, output='sos')
        self.consonant_state = np.zeros((1, 2))
        self.frozen = None
        self.bank_key = (bands, width)

    def carrier(self, count, settings):
        kind = settings.get('vocoder_carrier', 'saw')
        if kind not in CARRIERS:
            raise ValueError(f'Unknown vocoder carrier: {kind}')
        noise = self.rng.uniform(-1, 1, count)
        if kind == 'noise':
            return noise
        chord = settings.get('vocoder_chord', 'unison')
        if chord not in CHORDS:
            raise ValueError(f'Unknown vocoder chord: {chord}')
        note = np.clip(settings.get('vocoder_note', 45), 24, 96)
        detune = np.clip(settings.get('vocoder_detune', 6), 0, 50)
        width = np.clip(settings.get('vocoder_pulse_width', .5), .1, .9)
        signal = np.zeros(count)
        voice = 0
        # Two slightly detuned oscillators per chord note, with persistent phase.
        for interval in CHORDS[chord]:
            for cents in (-detune / 2, detune / 2):
                frequency = 440 * 2 ** ((note + interval - 69 + cents / 100) / 12)
                step = min(frequency / self.sr, .2)
                phase = (self.phases[voice] + step * np.arange(count)) % 1
                self.phases[voice] = (self.phases[voice] + step * count) % 1
                if kind == 'saw':
                    wave = 2 * phase - 1 - poly_blep(phase, step)
                else:
                    duty = .5 if kind == 'square' else width
                    wave = np.where(phase < duty, 1., -1.)
                    wave += poly_blep(phase, step) - poly_blep((phase - duty) % 1, step)
                    wave -= 2 * duty - 1  # Remove pulse-width-dependent DC offset.
                signal += wave
                voice += 1
        noise_mix = np.clip(settings.get('vocoder_noise', .08), 0, 1)
        return (1 - noise_mix) * signal / voice + noise_mix * noise

    def process(self, speech, settings):
        speech = np.asarray(speech, dtype=np.float64)
        bands = int(np.clip(round(settings.get('vocoder_bands', 24)), 8, 32))
        width = float(np.clip(settings.get('vocoder_bandwidth', 1), .6, 1.8))
        if self.bank_key != (bands, width):
            self.configure(bands, width)
        carrier = self.carrier(len(speech), settings)
        attack = np.exp(-1 / (self.sr * np.clip(settings.get('vocoder_attack_ms', 5), 1, 100) / 1000))
        release = np.exp(-1 / (self.sr * np.clip(settings.get('vocoder_release_ms', 70), 5, 500) / 1000))
        envelopes = np.empty((bands, len(speech)))
        carriers = np.empty_like(envelopes)
        power_pole = np.exp(-1 / (.03 * self.sr))
        for i, sos in enumerate(self.filters):
            analysis, self.analysis_state[i] = sosfilt(sos, speech, zi=self.analysis_state[i])
            envelopes[i], self.envelopes[i] = follow_envelope(analysis, self.envelopes[i], attack, release)
            band, self.synthesis_state[i] = sosfilt(sos, carrier, zi=self.synthesis_state[i])
            power, self.carrier_power[i] = lfilter([1 - power_pole], [1, -power_pole],
                                                 band * band, zi=self.carrier_power[i])
            # Normalize carrier bands, with a floor to avoid boosting near-empty
            # bands excessively when a sparse carrier has no harmonic there.
            carriers[i] = band / np.maximum(np.sqrt(np.maximum(power, 0)), .035)

        if settings.get('vocoder_freeze', False):
            # Freeze holds the last spectral envelope, not a loop of input audio.
            if self.frozen is None:
                self.frozen = envelopes[:, -1].copy()
            envelopes[:] = self.frozen[:, None]
        else:
            self.frozen = None

        # Shift the spectral envelope independently of carrier pitch. Interpolate
        # between analysis bands; zero outside the analyzed frequency range.
        shift = np.clip(settings.get('vocoder_formant', 0), -12, 12)
        position = np.arange(bands) - (shift / 12 * np.log(2) / np.log(self.centers[1] / self.centers[0]))
        left = np.floor(position).astype(int)
        fraction = position - left
        mapped = np.zeros_like(envelopes)
        for offset, weight in ((0, 1 - fraction), (1, fraction)):
            indices = left + offset
            valid = (indices >= 0) & (indices < bands)
            mapped[valid] += envelopes[indices[valid]] * weight[valid, None]
        result = np.sum(mapped * carriers, axis=0) * .65
        # Pass high-frequency speech detail to retain s, f and t consonants.
        consonants, self.consonant_state = sosfilt(self.consonant_filter, speech, zi=self.consonant_state)
        result += np.clip(settings.get('vocoder_consonants', .3), 0, 1) * consonants
        return result * 10 ** (np.clip(settings.get('vocoder_gain_db', 0), -18, 18) / 20)
