"""Signal-level checks for resynthesis, streaming state, and control behavior."""
import unittest
import numpy as np
from vocoder import Vocoder, VOCODER_DEFAULTS, CARRIERS
from voice_engine import VoiceProcessor, CHUNK, SAMPLERATE
from voice_config import DEFAULT_SETTINGS


class VocoderTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(CHUNK * 8) / SAMPLERATE
        # Speech-like harmonic input is intentionally distinct from carrier pitch.
        self.source = (.15 * np.sin(2 * np.pi * 220 * t) + .08 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
        self.settings = dict(VOCODER_DEFAULTS, vocoder_enabled=True, vocoder_noise=0, vocoder_consonants=0)

    def test_silent_input_does_not_leak_carrier(self):
        for carrier in CARRIERS:
            vocoder = Vocoder(seed=1)
            result = vocoder.process(np.zeros(CHUNK), dict(self.settings, vocoder_carrier=carrier))
            np.testing.assert_array_equal(result, 0)

    def test_streaming_matches_whole_signal(self):
        for carrier in CARRIERS:
            settings = dict(self.settings, vocoder_carrier=carrier, vocoder_noise=.1,
                            vocoder_formant=3, vocoder_chord='minor')
            full = Vocoder(seed=7).process(self.source, settings)
            processor = Vocoder(seed=7)
            blocks = np.concatenate([processor.process(block, settings) for block in np.split(self.source, 8)])
            np.testing.assert_allclose(blocks, full, atol=1e-8)

    def test_vocoder_audible_when_legacy_robot_mix_is_zero(self):
        settings = dict(DEFAULT_SETTINGS, enabled=True, vocoder_enabled=True, vocoder_mix=1)
        result = VoiceProcessor(seed=1).process(self.source, settings)
        self.assertGreater(float(np.sqrt(np.mean(result ** 2))), .005)
        self.assertGreater(float(np.mean(np.abs(result - self.source))), .02)
        self.assertTrue(np.isfinite(result).all())

    def test_zero_mix_and_disabled_are_bypasses(self):
        for overrides in (dict(vocoder_enabled=False), dict(vocoder_enabled=True, vocoder_mix=0)):
            settings = dict(DEFAULT_SETTINGS, enabled=True)
            settings.update(overrides)
            np.testing.assert_array_equal(VoiceProcessor().process(self.source, settings), self.source)

    def test_carrier_pitch_control_changes_spectrum(self):
        settings = dict(self.settings, vocoder_detune=0, vocoder_note=45)
        for note, expected in ((45, 110), (57, 220)):
            carrier = Vocoder().carrier(SAMPLERATE, dict(settings, vocoder_note=note))
            peak = int(np.argmax(np.abs(np.fft.rfft(carrier))))
            self.assertEqual(peak, expected)

    def test_release_decays_and_freeze_holds(self):
        normal = Vocoder(seed=1)
        frozen = Vocoder(seed=1)
        normal.process(self.source, self.settings)
        frozen.process(self.source, dict(self.settings, vocoder_freeze=True))
        silence = np.zeros(SAMPLERATE)
        tail = normal.process(silence, self.settings)
        held = frozen.process(silence, dict(self.settings, vocoder_freeze=True))
        self.assertLess(float(np.sqrt(np.mean(tail[-CHUNK:] ** 2))), .00001)
        self.assertGreater(float(np.sqrt(np.mean(held[-CHUNK:] ** 2))), .005)

    def test_band_and_formant_changes_are_finite(self):
        processor = Vocoder(seed=2)
        results = []
        for bands, formant in ((8, -12), (32, 12), (24, 0)):
            result = processor.process(self.source, dict(self.settings, vocoder_bands=bands, vocoder_formant=formant))
            self.assertTrue(np.isfinite(result).all())
            self.assertEqual(result.shape, self.source.shape)
            results.append(result)
        self.assertFalse(np.allclose(results[0], results[1]))


if __name__ == '__main__':
    unittest.main()
