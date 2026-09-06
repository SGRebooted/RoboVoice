"""Curated expansion pack. Run this file to add missing presets without overwriting saves."""
from voice_config import DEFAULT_SETTINGS, load_profiles, save_profiles


def build_library():
    speakers = load_profiles('speaker_types.json')
    result = {}

    def add(group, name, speaker='Clean', **changes):
        # Every entry is a complete snapshot. Keep levels moderate and freeze
        # disabled so selecting a fun voice cannot unexpectedly sustain a drone.
        values = dict(DEFAULT_SETTINGS)
        values.update(speakers[speaker])
        values.update(voice_type='Default', speaker_type=speaker,
                      vocoder_consonants=.45, vocoder_attack_ms=4,
                      vocoder_release_ms=75, repeat_interval=3,
                      repeat_strength=.45, gain=1)
        values.update(changes)
        result[f'{group} / {name}'] = values

    # Carrier pitch, chord, pulse shape and formants distinguish these robots.
    for name, note, carrier, chord, formant, mix in [
        ('Polite Android', 48, 'saw', 'unison', 0, .65),
        ('Iron Sentinel', 36, 'pulse', 'fifth', -2, .82),
        ('Pocket Droid', 57, 'square', 'unison', 2, .65),
        ('Starship Computer', 45, 'saw', 'unison', -1, .75),
        ('Chrome Diplomat', 50, 'pulse', 'major', 0, .6),
        ('Mining Automaton', 38, 'square', 'octave', -3, .78),
        ('Porcelain Servant', 55, 'saw', 'unison', 1, .55),
        ('Brass Butler', 43, 'pulse', 'fifth', -1, .7),
        ('Factory Foreman', 40, 'saw', 'unison', -2, .82),
        ('Crystal Intelligence', 60, 'pulse', 'minor', 3, .7),
    ]:
        add('Robot', name, 'Robot PA', vocoder_enabled=True, vocoder_note=note,
            vocoder_carrier=carrier, vocoder_chord=chord, vocoder_formant=formant,
            vocoder_mix=mix, vocoder_pulse_width=.4, vocoder_detune=5)

    # Mostly natural blends preserve acting and emotional inflections in dialogue.
    for name, pitch, formant, note, chord, mix in [
        ('Ancient Golem', -2, -3, 36, 'fifth', .5),
        ('Fey Messenger', 2, 3, 60, 'major', .35),
        ('Cursed Knight', -1, -2, 40, 'minor', .45),
        ('Crystal Oracle', 0, 2, 55, 'fifth', .55),
        ('Clockwork Familiar', 1, 1, 52, 'unison', .5),
        ('Dragon Herald', -3, -3, 38, 'octave', .4),
        ('Moonlit Spirit', 0, 3, 57, 'minor', .4),
        ('Dwarven Runekeeper', -2, -1, 43, 'unison', .35),
        ('Enchanted Armor', -1, 0, 45, 'fifth', .6),
        ('Talking Spellbook', 1, 2, 50, 'major', .3),
    ]:
        add('Fantasy', name, 'Clean', pitch_shift=pitch, robot_mix=.45,
            vocoder_enabled=True, vocoder_note=note, vocoder_formant=formant,
            vocoder_chord=chord, vocoder_mix=mix, vocoder_release_ms=90)

    for name, carrier, note, formant, mix, noise in [
        ('Airlock Whisper', 'noise', 45, 0, .6, .1),
        ('Nebula Navigator', 'saw', 47, 2, .65, .15),
        ('Deep Space Beacon', 'pulse', 38, -2, .8, .08),
        ('Alien Envoy', 'pulse', 53, 4, .65, .12),
        ('Ice Planet Surveyor', 'noise', 45, 3, .45, .1),
        ('Martian Archivist', 'square', 46, -1, .6, .18),
        ('Warp Core', 'saw', 35, -3, .75, .2),
        ('Asteroid Prospector', 'pulse', 42, 0, .55, .15),
        ('Lunar Broadcast', 'saw', 50, 1, .5, .1),
        ('Cosmic Interpreter', 'pulse', 55, -2, .7, .25),
    ]:
        add('Space', name, 'Radio', vocoder_enabled=True, vocoder_carrier=carrier,
            vocoder_note=note, vocoder_formant=formant, vocoder_mix=mix,
            vocoder_noise=noise, bitcrush_mix=.4, crackle=.015)

    # These suggest fictional devices; they do not emulate particular hardware.
    for name, rate, bits, crush, band, modulation in [
        ('Arcade Announcer', 16000, 8, .8, 0, 0),
        ('Handheld Hero', 11025, 8, .65, 500, 0),
        ('Pixel Shopkeeper', 16000, 10, .7, 800, 0),
        ('Dungeon Cartridge', 8000, 8, .55, 600, 0),
        ('Floppy Disk Wizard', 12000, 9, .75, 0, .15),
        ('Broken Intercom', 8000, 10, .6, 1000, .1),
        ('Toy Walkie Talkie', 11025, 9, .65, 1200, 0),
        ('Answering Machine', 16000, 12, .35, 700, 0),
        ('CRT Sidekick', 22050, 10, .7, 0, .2),
        ('Pocket Calculator', 8000, 7, .5, 900, .15),
    ]:
        add('Retro', name, 'Clean Compressor', sample_rate=rate, bit_depth=bits,
            bitcrush_mix=crush, bitcrush_drive_db=6, bandpass_freq=band,
            modulation_depth=modulation, modulation_rate=75, robot_mix=.7)

    for name, note, chord, waveform, release, formant in [
        ('Cathedral Circuit', 43, 'minor', 'saw', 150, -2),
        ('Celestial Assembly', 55, 'major', 'saw', 130, 2),
        ('Shadow Council', 38, 'minor', 'pulse', 120, -3),
        ('Prismatic Chorus', 52, 'major', 'pulse', 100, 3),
        ('Bronze Ensemble', 45, 'fifth', 'square', 95, 0),
        ('Holographic Quartet', 48, 'minor', 'saw', 85, 1),
        ('Neon Barbershop', 50, 'major', 'square', 80, 0),
        ('Frozen Hymn', 57, 'fifth', 'pulse', 170, 2),
        ('Subterranean Choir', 36, 'octave', 'saw', 125, -2),
        ('Astral Duet', 47, 'octave', 'pulse', 110, 1),
    ]:
        add('Choir', name, vocoder_enabled=True, vocoder_note=note,
            vocoder_chord=chord, vocoder_carrier=waveform, vocoder_release_ms=release,
            vocoder_formant=formant, vocoder_mix=.78, vocoder_detune=10)

    for name, note, chord, syllable, interval, strength, mix in [
        ('Nervous Service Bot', 52, 'unison', 180, 3.5, .35, .5),
        ('Corrupted Librarian', 43, 'minor', 220, 3, .5, .7),
        ('Sleepy Sentry', 38, 'unison', 280, 4, .4, .6),
        ('Excitable Companion', 55, 'major', 180, 2.8, .4, .45),
        ('Rusty Storyteller', 40, 'fifth', 220, 4.5, .35, .35),
        ('Malfunctioning Merchant', 48, 'unison', 200, 3, .45, .55),
        ('Damaged Battle Drone', 36, 'octave', 240, 3.5, .55, .75),
        ('Haunted Terminal', 45, 'minor', 260, 4, .45, .65),
        ('Stuttering Star Map', 50, 'fifth', 200, 3.8, .4, .6),
        ('Overworked Quest Giver', 47, 'unison', 180, 4.2, .3, .4),
    ]:
        add('Glitch', name, 'Glitchy Dialogue', mode='slightly_broken',
            vocoder_enabled=True, vocoder_note=note, vocoder_chord=chord,
            vocoder_mix=mix, syllable_ms=syllable, repeat_interval=interval,
            repeat_strength=strength, crackle=.02)
    return result


if __name__ == '__main__':
    presets = load_profiles('voice_presets.json')
    additions = build_library()
    count = 0
    for name, values in additions.items():
        if name not in presets:
            presets[name] = values
            count += 1
    save_profiles('voice_presets.json', presets)
    print(f'Added {count} presets; {len(presets)} total.')
