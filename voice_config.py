"""Complete preset snapshots and reusable voice/speaker configurations."""
import json
from pathlib import Path
from vocoder import VOCODER_DEFAULTS

ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS = dict(mode='normal', pitch_shift=0, distortion_amount=0,
                        hear_original=False, bandpass_freq=0, modulation_rate=90,
                        modulation_depth=0, bit_depth=16, bitcrush_mix=1,
                        bitcrush_drive_db=0, gain=1,
                        sample_rate=44100, compression_threshold_db=-24,
                        compression_ratio=1, makeup_db=0, crackle=0,
                        syllable_ms=220, robot_mix=0, repeat_interval=1.5,
                        repeat_strength=.95, repeat_threshold_db=-60)
DEFAULT_SETTINGS.update(VOCODER_DEFAULTS)
SPEAKER_KEYS = ('sample_rate', 'compression_threshold_db', 'compression_ratio',
                'makeup_db', 'bit_depth', 'bitcrush_mix', 'bitcrush_drive_db', 'crackle')
VOICE_KEYS = ('pitch_shift', 'bandpass_freq', 'modulation_rate', 'modulation_depth',
              'robot_mix') + tuple(VOCODER_DEFAULTS)


def load_profiles(filename):
    path = ROOT / filename
    if not path.exists():
        return {'Default': {}}
    with path.open(encoding='utf-8') as file:
        values = json.load(file)
    if not isinstance(values, dict) or any(not isinstance(v, dict) for v in values.values()):
        raise ValueError(f'{filename} must contain named configuration objects')
    return values or {'Default': {}}


def complete_preset(values, voices=None, speakers=None):
    # Older presets only saved four fields. Fill every missing field afresh,
    # rather than inheriting settings left behind by the last selected voice.
    result = dict(DEFAULT_SETTINGS)
    result.update((voices or {}).get(values.get('voice_type'), {}))
    result.update((speakers or {}).get(values.get('speaker_type'), {}))
    result.update(values)
    return result


def save_profiles(filename, values):
    # Replace atomically so an interrupted save cannot leave half a JSON file.
    path = ROOT / filename
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
