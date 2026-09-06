# RoboVoice



Live microphone voice changer for D&D. Uses the default microphone and output.

Use headphones to avoid microphone feedback.



## Setup



```powershell

python -m venv .venv

.\.venv\Scripts\python -m pip install -r requirements.txt

.\.venv\Scripts\python RoboVoice_Main.py

```



Enable Voice Changer applies effects; disabled passes through the microphone.

The effect controls scroll to fit smaller screens. Three gain/volume sliders remain
visible at the top:

- **Mic input gain** (0-400%): adjusts captured speech before detection,
  compression, and effects, including in bypass. This is software input trim,
  not the Windows microphone device's hardware level.
- **Overall output volume** (0-400%): master level for both local playback and
  the Python output queue, including when voice effects are disabled.
- **My playback volume** (0-400%): additional local speaker/headphone level.
  This does not change audio sent to another Python script. Zero mutes monitoring.

100% is unity gain. For example, 200% overall and 50% playback sends a doubled
signal to Python while retaining approximately the original local level, unless
limiting occurs. The input trim and both output routes limit peaks to avoid exceeding their output range;
large boosts can still flatten peaks and sound distorted. The volume settings
are session-level, start at 100% by default, and do not change with presets.
The separate Preset gain control remains saved with each voice.

Python callers can set initial levels with
`run_gui(output_queue=blocks, mic_gain=1.0, output_volume=2.0, playback_volume=0.5)`.
`monitor=False` always silences local playback regardless of its volume slider.



## Randomize and Glitchy Robot variations

**Randomize settings** chooses a restrained expressive, robot, choir, or retro
voice family and varies compatible controls. It avoids extreme pitch shifts,
heavy distortion, frozen vowels, and dense damage. Any random glitches use one
short repeat with a mild blend and a 3-5 second speaking interval. Mic input,
master volume, playback volume, and the current preset gain stay unchanged.
It also leaves the voice-changer enable checkbox alone.

**Undo randomize** restores the controls and selection labels from immediately
before the last randomization, including if you adjusted that result afterward.
Random results are marked Custom; use preset Save as to keep one. These limits
favor dialogue but cannot guarantee every combination suits every microphone
or voice.

Four additional Glitchy Robot presets have matching voice and speaker profiles:

- **Glitchy Robot - Expressive:** 35% vocoder blend leaves 65% natural voice,
  with light compression, fewer repeats, and less digital degradation. This
  retains more of your pitch inflection and volume dynamics; it does not detect
  emotions or make the fixed-pitch vocoder itself track expression.
- **Glitchy Robot - Gentle:** clearer robot voice with mild, occasional skips.
- **Glitchy Robot - Choir:** a layered minor-chord voice with spaced syllable repeats.
- **Glitchy Robot - Arcade:** pulse carrier and crunchy digital speaker texture.

The existing Glitchy Robot preset remains available. Choose Expressive first if
character acting and emotional delivery are more important than a fully
synthetic voice. The original lower Preset gain remains an output trim saved
with the voice; the top Mic input gain is separate and stays fixed across presets.

## Expansion preset library

The picker now contains 60 additional presets in six category submenus. Existing
presets remain at the top level. Categories are part of each saved preset name.

| Category | Voices to try |
| --- | --- |
| Robot | Polite Android, Iron Sentinel, Pocket Droid, Crystal Intelligence |
| Fantasy | Ancient Golem, Fey Messenger, Dragon Herald, Talking Spellbook |
| Space | Airlock Whisper, Alien Envoy, Warp Core, Cosmic Interpreter |
| Retro | Arcade Announcer, Dungeon Cartridge, CRT Sidekick, Pocket Calculator |
| Choir | Cathedral Circuit, Shadow Council, Neon Barbershop, Astral Duet |
| Glitch | Nervous Service Bot, Rusty Storyteller, Haunted Terminal, Overworked Quest Giver |

Each category contains ten voices. Fantasy voices blend more natural speech to
retain character acting; Choir voices emphasize layered synthetic harmonies;
Retro voices emphasize digital speaker texture. Glitch voices use a single,
spaced syllable repeat with a restrained blend. None enables vowel freeze.
All are full snapshots, so selections reset the complete effect configuration.
Your session mic and output volume controls remain untouched.

`preset_library.py` documents the curated settings and can add missing expansion
presets again without overwriting presets with existing names:
`.\.venv\Scripts\python preset_library.py`.
These are creative starting points; perceived clarity depends on your voice.

## Presets, voices, and speakers



- A **preset** restores every effect control, including speaker settings and

  original-voice mixing. Saving a preset captures the complete current sound.

  Enable/disable is deliberately separate. Manual edits mark the preset Custom.

- A **voice type** changes pitch, metallic carrier/depth, filter color, and robot

  mix, plus all vocoder controls. Other controls remain available for combining sounds.

- A **speaker type** changes dynamic compression, effective sample rate, bit

  depth, and crackle. Choose Clean, Robot PA, Radio, or Damaged Speaker; edit the

  controls and use Save as to create a reusable speaker profile.



Profiles are editable in `voice_types.json` and `speaker_types.json`. Complete

snapshots are saved in `voice_presets.json`. Existing presets and voice types

were backed up as `voice_presets.before-speakers.json` and

`voice_types.before-speakers.json` before this revision.



The original preset bug was partial restoration: the four saved parameters left

filtering/modulation from previous selections active. Missing fields in legacy

presets now receive defaults; complete presets restore their stored values.

A preset's stored values override any referenced voice or speaker profile.

Reapply an updated profile and save the preset to adopt profile changes.



## Sound controls



Metallic modulation uses a 20â€“200 Hz carrier to create a more robotic timbre.

Robot / effect mix controls the voice effect blend; Mix at least 50% original

voice imposes a 50% dry minimum. The speaker colors this combined signal, so

speaker settings also work when robot mix is zero.



Damaged modes replay 180-350 ms segments (220 ms normally, including Glitchy Robot). These approximate syllable duration; there is no syllable detector.

Slightly Broken repeats once, Extremely Damaged twice. Skipping affects the

whole voice independently of Robot mix and the original-voice checkbox.



Speech time between repeats controls automatic triggering (1.5 seconds normally,

2.5 seconds for Glitchy Robot). Only detected speech advances this timer; a short

continuous speech segment is required to avoid capturing silence. A cooldown

prevents events running together. Repeat strength controls how much live speech

is replaced; the default is 95%, so stutters are deliberately prominent. Glitchy Robot now uses
60%, keeping 40% of current speech audible through its two 220 ms repeats.
Speech during a replay does not advance the next event's timer.



With Voice Changer enabled, press Repeat syllable now while speaking to trigger

one event, even in Normal mode. A pending click waits for enough speech to fill

the selected length. The Repeats counter confirms events actually triggered.

Lower the speech detection threshold for a quiet microphone if necessary.

Normal mode has no automatic repeats. Disabling or changing modes clears replay.

Some live words are necessarily obscured by replacement; reduce repeat strength

or increase the interval for conversation-heavy sessions.



Speaker compression is dynamic range compression, not MP3-style file encoding:

threshold sets where compression begins, ratio sets its strength (1 bypasses),

and makeup gain restores volume. Sample rate is an **effective sound-quality

setting**, implemented with low-pass filtering and sample-and-hold. It does not

reopen the microphone at that hardware rate. Microphone capture, speaker output,

and the Python queue remain mono 44,100 Hz; pitch and duration stay unchanged.

Bit depth controls quantization, with a separate Bitcrush wet mix and compensated
input drive. Crackle adds adjustable sparse noise during

speech; set it to zero to remove this intentional texture.



The GUI counts audio stream status events as Audio dropouts and displays

processing errors. Unintended crackling can also arise from blockwise pitch

shifting; the counter alone cannot diagnose every artifact. Blocks are about

46 ms; device buffering and processing add latency. A microphone listening check

is still needed to judge clarity and character for your voice.



## Compression, bitcrushing, and console-inspired speakers

**Compression Only** evens out voice volume with full-rate, 16-bit output and no
vocoder or bitcrushing. Dynamic compression is often subtle: audio must exceed
the threshold before the ratio does anything. The top status line now displays
**Compression: X dB reduction**, averaged over the current block. Zero means no
gain reduction; lower the threshold if your microphone never reaches it.
Makeup gain restores level after compression; output volume controls loudness.

Bitcrushing runs after compression and effective sample-rate reduction:

- **Bitcrush depth:** fewer bits produce coarser digital steps; 16 bypasses.
- **Bitcrush wet mix:** 0 preserves the signal before quantization, 1 applies the
  complete crushed signal. A partial mix can keep quiet consonants audible.
- **Bitcrush input drive:** boosts into the quantizer then compensates the level
  afterward. It helps quiet input use more quantizer steps; high drive can clip
  the quantizer input and add distortion. It is separate from output volume.

Choose **SNES Voice** or **GBA Voice** for a complete console-inspired voice
without vocoding. Alternatively, apply the **SNES-inspired** or **GBA-inspired**
speaker type to any existing robot/vocoder voice. SNES-inspired uses a 16 kHz
quality setting, 9-bit quantization and an 80% crush mix; GBA-inspired uses
11.025 kHz, 8-bit quantization and a full crush mix. These are creative sound
profiles, not accurate emulations of either console's codecs or audio hardware.
Both still export mono 44.1 kHz audio through the Python queue.

Glitchy Robot uses the new **Glitchy Dialogue** speaker profile, with gentler
10-bit quantization and a 60% crush mix to retain quieter speech. Its vocoder
blend is also lighter. This addresses effects masking words; live microphone
or driver dropouts remain a separate possibility shown by the dropout counter.

Profiles before this change are saved in the three `*.before-console.json`
backup files. Save as can store your own compression/bitcrush speaker settings.

## Receive audio in another Python file



Run `consume_voice.py` and replace its `handle_audio(samples, sample_rate)` with

your own handling. It imports `run_gui(output_queue=blocks, monitor=True)` and

consumes a bounded queue on a worker thread. Tk runs on the main thread.

Set `monitor=False` to receive audio without speaker playback.



Each queue item is a copied mono NumPy float32 array at 44,100 Hz, matching the

speaker output (including bypass when disabled). A full queue drops the oldest

waiting block so slow consumers cannot stall live audio. This is not guaranteed

lossless recording. Slow file or network work belongs in the consumer thread.

Both files run in one Python process; this does not create a virtual microphone

or connect to an independently launched process.



For offline processing import `VoiceProcessor` from `voice_engine` and call

`process(samples, settings)` with successive mono blocks at 44,100 Hz.



## Checks



Run `.\.venv\Scripts\python -m unittest discover -s tests`.

Signal tests do not establish perceived speech quality.

