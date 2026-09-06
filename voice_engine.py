"""Speech-first processing, independent of Tk and microphone hardware."""
import numpy as np
from scipy.signal import butter, sosfilt, lfilter
from vocoder import Vocoder

SAMPLERATE = 44100
CHUNK = 2048  # About 46 ms; the old 8192-sample block was about 186 ms.


def apply_volume(samples, volume=1., ceiling=.98):
    """Scale audio without mutating a block shared with another output route."""
    return np.clip(np.asarray(samples) * np.clip(volume, 0, 4),
                   -ceiling, ceiling).astype(np.float32)


class VoiceProcessor:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.vocoder = Vocoder(SAMPLERATE, seed)
        self.vocoder_active = False
        self.phase = 0.0
        self.filter_key = None
        self.history = np.empty(0, dtype=np.float32)
        self.replay = np.empty(0, dtype=np.float32)
        self.replay_blend = np.empty(0, dtype=np.float32)
        self.cooldown = 0
        self.mode = "normal"
        self.voiced_samples = 0
        self.speech_run = 0
        self.repeat_events = 0
        self.trigger_token = 0
        self.manual_pending = False
        self.envelope = np.zeros(1)
        self.gain_state = np.ones(1)
        self.sample_key = None
        self.sample_clock = 0.0
        self.held_sample = 0.0
        self.compression_reduction_db = 0.0

    def process(self, samples, settings):
        dry = np.nan_to_num(np.asarray(samples, dtype=np.float32)).copy()
        # Mic trim precedes speech detection, compression and all voice effects.
        # This is session-level input gain, separate from the saved output trim.
        dry = apply_volume(dry, settings.get('mic_gain', 1), ceiling=1)
        # Disabled bypasses voice effects; session mic/master levels still apply.
        if not settings.get("enabled", False):
            self.history = self.replay = np.empty(0, dtype=np.float32)
            self.replay_blend = np.empty(0, dtype=np.float32)
            self.filter_key = None
            self.phase = 0.0
            self.cooldown = 0
            self.envelope[:] = 0
            self.compression_reduction_db = 0.0
            self.gain_state[:] = 1
            self.sample_key = None
            self.voiced_samples = 0
            self.speech_run = 0
            self.manual_pending = False
            self.trigger_token = settings.get('repeat_trigger', 0)
            if self.vocoder_active:
                self.vocoder.reset()
                self.vocoder_active = False
            # Global output volume also controls unmodified microphone audio.
            # Preset gain remains bypassed when the voice changer is disabled.
            return apply_volume(dry, settings.get('output_volume', 1), ceiling=1)

        wet = dry.copy()
        pitch = settings.get("pitch_shift", 0)
        if pitch:
            import librosa
            wet = librosa.effects.pitch_shift(wet, sr=SAMPLERATE, n_steps=pitch)

        center = settings.get("bandpass_freq", 600)
        if center > 0:
            # Keep consonants: a broad filtered voice colors only part of the mix.
            if self.filter_key != center:
                self.sos = butter(2, [max(80, center - 600), min(8000, center + 3500)],
                                  btype="bandpass", fs=SAMPLERATE, output="sos")
                self.zi = np.zeros((len(self.sos), 2))
                self.filter_key = center
            filtered, self.zi = sosfilt(self.sos, wet, zi=self.zi)
            wet = 0.65 * wet + 0.35 * filtered
        else:
            self.filter_key = None

        rate = settings.get("modulation_rate", 0)
        depth = np.clip(settings.get("modulation_depth", 0), 0, 1)
        # Carry phase across blocks so modulation does not jump at each boundary.
        phase = self.phase + 2 * np.pi * rate * np.arange(len(wet)) / SAMPLERATE
        # Audio-rate ring modulation gives a metallic robot tone. The previous
        # 1–20 Hz shallow modulation mostly sounded like ordinary tremolo.
        wet *= (1 - depth) + depth * np.cos(phase)
        self.phase = (self.phase + 2 * np.pi * rate * len(wet) / SAMPLERATE) % (2 * np.pi)

        amount = np.clip(settings.get("distortion_amount", 0), 0, 100) / 100
        if amount:
            # Parallel distortion retains speech transients instead of replacing them.
            drive = 1 + 5 * amount
            wet = (1 - 0.4 * amount) * wet + 0.4 * amount * np.tanh(wet * drive) / drive
        # Robot mix is explicit now, instead of an unavoidable 25–50% dry blend.
        robot_mix = np.clip(settings.get('robot_mix', 0.85), 0, 1)
        dry_mix = max(1 - robot_mix, 0.5 if settings.get("hear_original", False) else 0)
        # The speaker colors the complete voice, including the dry intelligibility
        # blend. Choosing a speaker therefore also works with robot mix set to zero.
        mixed = (1 - dry_mix) * wet + dry_mix * dry
        # Vocoder has its own mix so it remains audible with the legacy Robot
        # mix at zero. Both the GUI and Python output share this exact pipeline.
        vocoder_mix = np.clip(settings.get('vocoder_mix', .85), 0, 1)
        if settings.get('vocoder_enabled', False) and vocoder_mix > 0:
            synthesized = self.vocoder.process(mixed, settings)
            mixed = (1 - vocoder_mix) * mixed + vocoder_mix * synthesized
            self.vocoder_active = True
        elif self.vocoder_active:
            self.vocoder.reset()
            self.vocoder_active = False
        # Damage is a timing effect on the whole voice, independent of robot mix.
        mixed = self.skip_syllable(mixed, dry, settings)
        result = self.speaker_input(mixed, settings)
        # Optional texture is independent of device dropouts or processing defects.
        crackle = np.clip(settings.get('crackle', 0), 0, 1)
        if crackle:
            impulses = self.rng.random(len(result)) < 35 / SAMPLERATE
            result += impulses * self.rng.uniform(-1, 1, len(result)) * crackle * min(1, np.max(np.abs(dry)) * 8)
        result *= settings.get('gain', 1)
        # Apply master volume before limiting, so quiet voices can be boosted.
        # The returned block feeds both playback and the Python output queue.
        return apply_volume(np.nan_to_num(result), settings.get('output_volume', 1))

    def skip_syllable(self, audio, microphone, settings):
        """Replay a syllable snapshot with bounded, speech-driven triggering."""
        mode = settings.get('mode', 'normal')
        if mode != self.mode:
            self.history = self.replay = self.replay_blend = np.empty(0, dtype=np.float32)
            self.voiced_samples = 0
            self.speech_run = 0
            self.cooldown = 0
            self.manual_pending = False
            self.mode = mode
        token = settings.get('repeat_trigger', 0)
        if token != self.trigger_token:
            self.manual_pending = True
            self.trigger_token = token

        # round avoids the old 350 ms capacity being one sample shorter than
        # the requested replay at the maximum slider setting.
        length = round(SAMPLERATE * np.clip(settings.get('syllable_ms', 220), 180, 350) / 1000)
        self.history = np.concatenate((self.history, audio))[-round(.35 * SAMPLERATE):]
        threshold = 10 ** (np.clip(settings.get('repeat_threshold_db', -60), -80, -20) / 20)
        voiced = np.sqrt(np.mean(microphone.astype(float) ** 2)) >= threshold
        if voiced:
            # Do not count words spoken during a replay toward the next event.
            # Otherwise long repeats consume most of the advertised interval.
            if not len(self.replay):
                self.voiced_samples += len(audio)
            self.speech_run += len(audio)
        else:
            # A pause cannot fill the speech timer or supply a silent syllable.
            self.speech_run = 0
        self.cooldown = max(0, self.cooldown - len(audio))
        damaged = mode == 'extremely_damaged'
        automatic = mode in ('slightly_broken', 'extremely_damaged')
        interval = max(length, round(SAMPLERATE * settings.get('repeat_interval', 1.5)))
        due = self.manual_pending or (automatic and self.voiced_samples >= interval)
        ready = len(self.history) >= length and self.speech_run >= length
        if not len(self.replay) and ready and due and (self.manual_pending or self.cooldown == 0):
            syllable = self.history[-length:].copy()
            fade = min(220, length // 2)
            syllable[:fade] *= np.linspace(0, 1, fade)
            syllable[-fade:] *= np.linspace(1, 0, fade)
            self.replay = np.tile(syllable, 2 if damaged else 1)
            # Crossfade only the entry and exit of the event. High strength now
            # creates an audible skip rather than a faint echo behind live words.
            strength = np.clip(settings.get('repeat_strength', .95), 0, 1)
            self.replay_blend = np.full(len(self.replay), strength, dtype=np.float32)
            self.replay_blend[:fade] *= np.linspace(0, 1, fade)
            self.replay_blend[-fade:] *= np.linspace(1, 0, fade)
            self.cooldown = len(self.replay) + round(.5 * SAMPLERATE)
            self.voiced_samples = 0
            self.speech_run = 0
            self.manual_pending = False
            self.repeat_events += 1
        n = min(len(audio), len(self.replay))
        if n:
            blend = self.replay_blend[:n]
            audio[:n] = (1 - blend) * audio[:n] + blend * self.replay[:n]
            self.replay = self.replay[n:]
            self.replay_blend = self.replay_blend[n:]
        return audio

    def speaker_input(self, data, settings):
        """Color microphone-derived audio without changing stream rate or timing."""
        # Smooth power and gain continuously to avoid gain jumps at block edges.
        pole = np.exp(-1 / (0.01 * SAMPLERATE))
        power, self.envelope = lfilter([1 - pole], [1, -pole], data.astype(float) ** 2, zi=self.envelope)
        db = 10 * np.log10(np.maximum(power, 1e-12))
        ratio = np.clip(settings.get('compression_ratio', 1), 1, 20)
        threshold = np.clip(settings.get('compression_threshold_db', -24), -60, 0)
        reduction = np.maximum(db - threshold, 0) * (1 - 1 / ratio)
        self.compression_reduction_db = float(np.mean(reduction))
        target = 10 ** (-reduction / 20)
        gain, self.gain_state = lfilter([1 - pole], [1, -pole], target, zi=self.gain_state)
        # Ratio 1 really bypasses compression, including the smoothing startup.
        audio = data * (gain if ratio > 1 else 1) * 10 ** (np.clip(settings.get('makeup_db', 0), 0, 18) / 20)
        rate = int(np.clip(settings.get('sample_rate', SAMPLERATE), 4000, SAMPLERATE))
        if rate != self.sample_key:
            self.sample_key = rate
            self.sample_clock = 0.0
            self.held_sample = 0.0
            self.sample_sos = butter(4, min(rate * .45, 20000), fs=SAMPLERATE, output='sos')
            self.sample_zi = np.zeros((len(self.sample_sos), 2))
        if rate < SAMPLERATE:
            # Low-pass before sample-and-hold to control aliasing. Keep the sample
            # clock between blocks, including non-integer ratios such as 8000 Hz.
            audio, self.sample_zi = sosfilt(self.sample_sos, audio, zi=self.sample_zi)
            held = np.empty_like(audio)
            for i, value in enumerate(audio):
                if self.sample_clock <= 0:
                    self.held_sample = value
                    self.sample_clock += SAMPLERATE / rate
                held[i] = self.held_sample
                self.sample_clock -= 1
            audio = held
        bits = int(np.clip(settings.get('bit_depth', 16), 4, 16))
        mix = np.clip(settings.get('bitcrush_mix', 1), 0, 1)
        if bits < 16 and mix > 0:
            # Compression evens out level; quantization provides the digital
            # texture. Drive fills more quantizer steps for quiet microphones,
            # then is compensated so it is not another output-volume control.
            drive = 10 ** (np.clip(settings.get('bitcrush_drive_db', 0), 0, 24) / 20)
            levels = 2 ** (bits - 1)
            crushed = np.round(np.clip(audio * drive, -1, 1) * levels) / levels / drive
            audio = (1 - mix) * audio + mix * crushed
        return audio
