"""Random voices built from restrained sound families rather than every slider."""
import random
from voice_config import DEFAULT_SETTINGS


def randomize_settings(current, rng=None):
    rng = rng or random.Random()
    result = dict(DEFAULT_SETTINGS)
    # Keep the user's level trim. Mic/master/playback levels live outside presets.
    result['gain'] = current.get('gain', 1)
    family = rng.choice(('expressive', 'robot', 'choir', 'retro'))
    result.update(bit_depth=16, bitcrush_mix=.5, compression_threshold_db=-26,
                  compression_ratio=2, makeup_db=2, vocoder_freeze=False)
    if family == 'retro':
        result.update(sample_rate=rng.choice((11025, 16000, 22050)),
                      bit_depth=rng.choice((8, 10, 12)),
                      bitcrush_drive_db=6, bitcrush_mix=rng.uniform(.35, .7))
    else:
        result.update(vocoder_enabled=True, vocoder_note=rng.choice((40, 43, 45, 48, 50)),
                      vocoder_carrier=rng.choice(('saw', 'pulse')),
                      vocoder_bands=rng.choice((20, 24, 28)),
                      vocoder_pulse_width=rng.uniform(.35, .6),
                      vocoder_attack_ms=rng.uniform(3, 8),
                      vocoder_release_ms=rng.uniform(45, 100),
                      vocoder_formant=rng.uniform(-2, 2),
                      vocoder_consonants=rng.uniform(.35, .55),
                      vocoder_noise=rng.uniform(.05, .12),
                      vocoder_detune=rng.uniform(2, 12))
        # Expression survives in the natural component; fixed synth notes do
        # not track emotional pitch inflections in the original voice.
        result['vocoder_mix'] = rng.uniform(.3, .5) if family == 'expressive' else rng.uniform(.65, .85)
        if family == 'choir':
            result['vocoder_chord'] = rng.choice(('fifth', 'minor', 'major'))
    if rng.random() < .3:
        result.update(mode='slightly_broken', repeat_interval=rng.uniform(3, 5),
                      repeat_strength=rng.uniform(.35, .55),
                      syllable_ms=rng.choice((180, 200, 220)))
    # Never randomly combine extreme distortion, heavy damage, pitch shifting,
    # frozen vowels, and aggressive quantization. These ranges favor dialogue.
    return result
