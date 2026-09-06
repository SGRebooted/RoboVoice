import json
from pathlib import Path
import unittest
from queue import Queue
import numpy as np
from voice_engine import VoiceProcessor, CHUNK, SAMPLERATE
from RoboVoice_Main import publish_audio, playback_audio
from voice_config import DEFAULT_SETTINGS, complete_preset, load_profiles


class AlwaysTrigger:
    def random(self):
        return 0


class VoiceTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(CHUNK) / SAMPLERATE
        self.speech = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def test_disabled_is_exact_bypass(self):
        actual = VoiceProcessor().process(self.speech, dict(enabled=False, gain=3, pitch_shift=-12))
        np.testing.assert_array_equal(actual, self.speech)

    def test_master_volume_scales_enabled_and_bypassed_audio(self):
        for enabled in (False, True):
            settings = dict(DEFAULT_SETTINGS, enabled=enabled, output_volume=2)
            result = VoiceProcessor().process(self.speech, settings)
            np.testing.assert_allclose(result, self.speech * 2, atol=1e-7)
            settings['output_volume'] = 0
            np.testing.assert_array_equal(VoiceProcessor().process(self.speech, settings), 0)

    def test_playback_volume_is_independent_of_exported_master_audio(self):
        settings = dict(DEFAULT_SETTINGS, enabled=True, output_volume=2)
        master = VoiceProcessor().process(self.speech, settings)
        playback = playback_audio(master, .5)
        blocks = Queue(maxsize=1)
        publish_audio(blocks, master)
        np.testing.assert_allclose(playback, self.speech, atol=1e-7)
        np.testing.assert_allclose(blocks.get_nowait(), self.speech * 2, atol=1e-7)
        np.testing.assert_array_equal(playback_audio(master, 0), 0)
        np.testing.assert_array_equal(playback_audio(master, 4, monitor=False), 0)
        np.testing.assert_allclose(master, self.speech * 2, atol=1e-7)

    def test_volume_boost_limits_overload(self):
        settings = dict(DEFAULT_SETTINGS, enabled=True, output_volume=4)
        result = VoiceProcessor().process(np.full(CHUNK, .6, dtype=np.float32), settings)
        self.assertLessEqual(float(np.max(np.abs(result))), .981)
        self.assertLessEqual(float(np.max(np.abs(playback_audio(result, 4)))), .981)

    def test_legacy_robot_presets_use_matching_vocoder_voices(self):
        presets = load_profiles('voice_presets.json')
        voices = load_profiles('voice_types.json')
        from voice_config import VOICE_KEYS
        for name in ('G1 Soundwave', 'HK-74', 'AUTO (Wall-E)', 'Cyber Warforged', 'Glitchy Robot'):
            self.assertTrue(presets[name]['vocoder_enabled'])
            for key in VOICE_KEYS:
                self.assertEqual(presets[name][key], voices[name][key])
        self.assertEqual(presets['G1 Soundwave']['vocoder_chord'], presets['Robot Choir']['vocoder_chord'])
        self.assertFalse(presets['Default']['vocoder_enabled'])

    def test_slow_consumer_receives_latest_independent_block(self):
        blocks = Queue(maxsize=1)
        publish_audio(blocks, np.zeros(CHUNK, dtype=np.float32))
        publish_audio(blocks, self.speech)
        received = blocks.get_nowait()
        np.testing.assert_array_equal(received, self.speech)
        self.assertFalse(np.shares_memory(received, self.speech))

    def test_damage_keeps_live_speech_after_echo_ends(self):
        processor = VoiceProcessor()
        processor.rng = AlwaysTrigger()
        settings = dict(enabled=True, mode='extremely_damaged', bandpass_freq=0, repeat_trigger=1)
        for _ in range(20):
            result = processor.process(self.speech, settings)
        # A pair of 220 ms repeats finishes before the cooldown permits another.
        self.assertEqual(len(processor.replay), 0)
        np.testing.assert_allclose(result, self.speech, atol=1e-7)

    def test_damage_captures_a_whole_syllable(self):
        processor = VoiceProcessor()
        processor.rng = AlwaysTrigger()
        settings = dict(enabled=True, mode='extremely_damaged', bandpass_freq=0, syllable_ms=280, repeat_trigger=1)
        for _ in range(7):
            processor.process(self.speech, settings)
        self.assertEqual(len(processor.replay), 2 * int(.280 * SAMPLERATE) - CHUNK)

    def test_skipping_audible_with_default_zero_robot_mix(self):
        processor = VoiceProcessor()
        settings = dict(DEFAULT_SETTINGS, enabled=True, mode='slightly_broken',
                        repeat_interval=.5, repeat_strength=1)
        rng = np.random.default_rng(8)
        differences = []
        for _ in range(18):
            # Changing input is essential: replaying a steady sine hides timing bugs.
            block = rng.normal(0, .06, CHUNK).astype(np.float32)
            output = processor.process(block, settings)
            differences.append(float(np.mean(np.abs(output - block))))
        self.assertEqual(processor.repeat_events, 1)
        self.assertGreater(max(differences), .03)

    def test_maximum_syllable_length_can_trigger(self):
        processor = VoiceProcessor()
        settings = dict(DEFAULT_SETTINGS, enabled=True, syllable_ms=350, repeat_trigger=1)
        for _ in range(8):
            processor.process(self.speech, settings)
        self.assertEqual(processor.repeat_events, 1)
        self.assertEqual(len(processor.replay), round(.35 * SAMPLERATE) - CHUNK)

    def test_silence_does_not_trigger_automatic_replay(self):
        processor = VoiceProcessor()
        settings = dict(DEFAULT_SETTINGS, enabled=True, mode='extremely_damaged', repeat_interval=.5)
        for _ in range(50):
            result = processor.process(np.zeros(CHUNK, dtype=np.float32), settings)
        self.assertEqual(processor.repeat_events, 0)
        self.assertFalse(np.any(result))

    def test_manual_repeat_works_in_normal_mode_once_per_click(self):
        processor = VoiceProcessor()
        settings = dict(DEFAULT_SETTINGS, enabled=True, repeat_trigger=1)
        for _ in range(50):
            processor.process(self.speech, settings)
        self.assertEqual(processor.repeat_events, 1)

    def test_speaker_rate_and_compression_are_continuous(self):
        settings = dict(enabled=True, bandpass_freq=0, sample_rate=8000,
                        compression_ratio=4, makeup_db=4)
        audio = np.tile(self.speech, 10)
        whole = VoiceProcessor().process(audio, settings)
        processor = VoiceProcessor()
        split = np.concatenate([processor.process(block, settings) for block in np.split(audio, 10)])
        np.testing.assert_allclose(split, whole, atol=1e-6)

    def test_compressor_reduces_loud_signal(self):
        audio = np.full(SAMPLERATE, .5, dtype=np.float32)
        settings = dict(enabled=True, bandpass_freq=0, robot_mix=0,
                        compression_ratio=4, compression_threshold_db=-24)
        result = VoiceProcessor().process(audio, settings)
        self.assertLess(float(np.mean(result[-1000:])), .15)
        self.assertGreater(float(np.mean(result[-1000:])), .08)

    def test_compression_meter_distinguishes_below_and_above_threshold(self):
        processor = VoiceProcessor()
        settings = dict(DEFAULT_SETTINGS, enabled=True, compression_ratio=4)
        processor.process(np.full(SAMPLERATE, .001, dtype=np.float32), settings)
        self.assertEqual(processor.compression_reduction_db, 0)
        processor.process(np.full(SAMPLERATE, .5, dtype=np.float32), settings)
        self.assertGreater(processor.compression_reduction_db, 10)
        processor.process(self.speech, dict(settings, enabled=False))
        self.assertEqual(processor.compression_reduction_db, 0)

    def test_bitcrush_mix_and_drive_preserve_quiet_speech(self):
        quiet = np.full(CHUNK, .01, dtype=np.float32)
        settings = dict(DEFAULT_SETTINGS, enabled=True, bit_depth=4)
        crushed = VoiceProcessor().process(quiet, settings)
        np.testing.assert_array_equal(crushed, 0)
        blended = VoiceProcessor().process(quiet, dict(settings, bitcrush_mix=.5))
        np.testing.assert_allclose(blended, quiet * .5)
        driven = VoiceProcessor().process(quiet, dict(settings, bitcrush_drive_db=24))
        self.assertGreater(float(np.mean(driven)), .005)
        np.testing.assert_array_equal(VoiceProcessor().process(quiet, dict(settings, bitcrush_mix=0)), quiet)

    def test_replay_does_not_advance_next_event_timer(self):
        processor = VoiceProcessor()
        settings = dict(DEFAULT_SETTINGS, enabled=True, mode='extremely_damaged', repeat_trigger=1)
        for _ in range(5):
            processor.process(self.speech, settings)
        self.assertEqual(processor.repeat_events, 1)
        self.assertGreater(len(processor.replay), 0)
        processor.process(self.speech, settings)
        self.assertEqual(processor.voiced_samples, 0)

    def test_glitchy_dialogue_retains_live_words_and_spaces_repeats(self):
        settings = dict(load_profiles('voice_presets.json')['Glitchy Robot'], enabled=True)
        processor = VoiceProcessor()
        old = np.full(CHUNK, .1, dtype=np.float32)
        for _ in range(55):
            processor.skip_syllable(old.copy(), old, settings)
        self.assertEqual(processor.repeat_events, 1)
        # During a repeat of a positive signal, the new negative word must still
        # contribute 40%, not the 5% retained by the previous configuration.
        incoming = np.full(CHUNK, -.1, dtype=np.float32)
        result = processor.skip_syllable(incoming.copy(), incoming, settings)
        np.testing.assert_allclose(result, .02, atol=1e-6)
        for _ in range(30):
            processor.skip_syllable(old.copy(), old, settings)
        self.assertEqual(processor.repeat_events, 1)

    def test_low_sample_rate_attenuates_high_frequencies(self):
        audio = (.2 * np.sin(2 * np.pi * 12000 * np.arange(SAMPLERATE) / SAMPLERATE)).astype(np.float32)
        result = VoiceProcessor().process(audio, dict(enabled=True, robot_mix=0, sample_rate=8000))
        self.assertLess(float(np.sqrt(np.mean(result[1000:] ** 2))), .01)

    def test_old_presets_reset_all_missing_settings(self):
        result = complete_preset(dict(pitch_shift=-3))
        self.assertEqual(result['pitch_shift'], -3)
        for key, value in DEFAULT_SETTINGS.items():
            if key != 'pitch_shift':
                self.assertEqual(result[key], value)

    def test_builtin_presets_store_all_controls_and_valid_profiles(self):
        presets = load_profiles('voice_presets.json')
        voices = load_profiles('voice_types.json')
        speakers = load_profiles('speaker_types.json')
        for name, values in presets.items():
            with self.subTest(preset=name):
                self.assertTrue(DEFAULT_SETTINGS.keys() <= values.keys())
                self.assertIn(values['voice_type'], voices)
                self.assertIn(values['speaker_type'], speakers)

    def test_filter_and_modulation_are_continuous_across_blocks(self):
        settings = dict(enabled=True, bandpass_freq=850, modulation_rate=8, modulation_depth=.25)
        audio = np.tile(self.speech, 3)
        whole = VoiceProcessor().process(audio, settings)
        processor = VoiceProcessor()
        blocks = np.concatenate([processor.process(block, settings) for block in np.split(audio, 3)])
        np.testing.assert_allclose(blocks, whole, atol=1e-6)

    def test_presets_produce_valid_audio(self):
        presets = json.loads(Path(__file__).resolve().parents[1].joinpath('voice_presets.json').read_text())
        for name, settings in presets.items():
            with self.subTest(preset=name):
                result = VoiceProcessor(seed=0).process(self.speech, dict(settings, enabled=True))
                self.assertEqual(result.shape, self.speech.shape)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
                self.assertLessEqual(float(np.max(np.abs(result))), .981)


if __name__ == '__main__':
    unittest.main()
