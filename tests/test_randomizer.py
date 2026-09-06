import random
import unittest
import numpy as np
from voice_config import DEFAULT_SETTINGS, load_profiles
from voice_randomizer import randomize_settings
from voice_engine import VoiceProcessor, CHUNK


class RandomizerTests(unittest.TestCase):
    def test_randomizer_retains_trim_and_bounds_destructive_combinations(self):
        results = []
        for seed in range(100):
            values = randomize_settings(dict(DEFAULT_SETTINGS, gain=1.7), random.Random(seed))
            self.assertEqual(values['gain'], 1.7)
            self.assertFalse(values['vocoder_freeze'])
            self.assertEqual(values['pitch_shift'], 0)
            self.assertNotEqual(values['mode'], 'extremely_damaged')
            self.assertGreaterEqual(values['bit_depth'], 8)
            self.assertNotIn('mic_gain', values)
            self.assertNotIn('output_volume', values)
            self.assertNotIn('playback_volume', values)
            if values['mode'] != 'normal':
                self.assertLessEqual(values['repeat_strength'], .55)
                self.assertGreaterEqual(values['repeat_interval'], 3)
            results.append(tuple(sorted(values.items())))
        self.assertGreater(len(set(results)), 90)

    def test_randomized_voices_make_finite_nonzero_audio(self):
        source = np.random.default_rng(1).normal(0, .06, CHUNK).astype(np.float32)
        for seed in range(20):
            settings = randomize_settings(DEFAULT_SETTINGS, random.Random(seed))
            result = VoiceProcessor(seed=1).process(source, dict(settings, enabled=True))
            self.assertTrue(np.isfinite(result).all())
            self.assertGreater(float(np.sqrt(np.mean(result ** 2))), .001)
            self.assertLessEqual(float(np.max(np.abs(result))), .981)

    def test_mic_gain_applies_before_compression_and_in_bypass(self):
        source = np.full(CHUNK, .02, dtype=np.float32)
        for enabled in (False, True):
            settings = dict(DEFAULT_SETTINGS, enabled=enabled, mic_gain=2)
            np.testing.assert_allclose(VoiceProcessor().process(source, settings), source * 2)
        processor = VoiceProcessor()
        processor.process(np.full(44100, .02, dtype=np.float32),
                          dict(DEFAULT_SETTINGS, enabled=True, mic_gain=4, compression_ratio=4))
        self.assertGreater(processor.compression_reduction_db, 0)

    def test_expressive_variant_preserves_more_natural_voice_and_dynamics(self):
        presets = load_profiles('voice_presets.json')
        base = presets['Glitchy Robot']
        expressive = presets['Glitchy Robot - Expressive']
        self.assertLess(expressive['vocoder_mix'], .5)
        self.assertLess(expressive['compression_ratio'], base['compression_ratio'])
        self.assertLess(expressive['repeat_strength'], base['repeat_strength'])
        self.assertGreater(expressive['repeat_interval'], base['repeat_interval'])
