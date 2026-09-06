"""RoboVoice controls and queue output; audio processing lives in voice_engine."""
import tkinter as tk
from tkinter import ttk
from queue import Full, Empty
import numpy as np
import sounddevice as sd
from voice_engine import VoiceProcessor, SAMPLERATE, CHUNK, apply_volume
from vocoder import CARRIERS, CHORDS, follow_envelope
from voice_randomizer import randomize_settings
from voice_config import (DEFAULT_SETTINGS, SPEAKER_KEYS, VOICE_KEYS,
                          load_profiles, complete_preset, save_profiles)


def publish_audio(output_queue, processed):
    """Publish an independent block; a slow consumer must never stall audio."""
    try:
        output_queue.put_nowait(processed.copy())
    except Full:
        try:
            output_queue.get_nowait()
        except Empty:
            pass
        try:
            output_queue.put_nowait(processed.copy())
        except Full:
            pass


def playback_audio(processed, volume=1., monitor=True):
    """Local monitoring volume never changes audio exported to another script."""
    return apply_volume(processed, volume) if monitor else np.zeros_like(processed)


def run_gui(output_queue=None, monitor=True, output_volume=1., playback_volume=1., mic_gain=1.):
    """Run Tk on the main thread; optionally send 44.1 kHz mono blocks to a Queue.

    Volume arguments are linear multipliers (0-4). Output volume affects both
    destinations; playback volume is additional local monitoring gain only.
    Mic gain trims the input before detection and effects. All three are session
    controls, independent of saved voice/preset settings.
    """
    presets = load_profiles('voice_presets.json')
    voices = load_profiles('voice_types.json')
    speakers = load_profiles('speaker_types.json')
    root = tk.Tk()
    root.title('RoboVoice')
    root.geometry('570x850')
    enabled = tk.BooleanVar(value=False)
    ttk.Checkbutton(root, text='Enable Voice Changer', variable=enabled).pack(anchor='w', padx=12, pady=8)
    status_text = tk.StringVar(value='Disabled — microphone passthrough')
    ttk.Label(root, textvariable=status_text, wraplength=530).pack(fill='x', padx=12)

    # Session-level volume controls stay visible and do not jump on preset changes.
    # The master affects every destination; playback is an additional local trim.
    master_var = tk.DoubleVar(value=np.clip(output_volume, 0, 4) * 100)
    playback_var = tk.DoubleVar(value=np.clip(playback_volume, 0, 4) * 100)
    mic_var = tk.DoubleVar(value=np.clip(mic_gain, 0, 4) * 100)
    for label, variable in [('Mic input gain (before effects)', mic_var),
                             ('Overall output volume (all destinations)', master_var),
                             ('My playback volume (speakers / headphones)', playback_var)]:
        line = ttk.Frame(root, padding=(12, 3))
        line.pack(fill='x')
        ttk.Label(line, text=label).pack(side='left')
        readout = ttk.Label(line)
        readout.pack(side='right')
        def show_volume(*args, variable=variable, readout=readout):
            readout.config(text=f'{variable.get():.0f}%')
        variable.trace_add('write', show_volume)
        show_volume()
        ttk.Scale(root, from_=0, to=400, variable=variable).pack(fill='x', padx=12)

    # Scrolling keeps the additional speaker controls accessible on small screens.
    container = ttk.Frame(root)
    container.pack(fill='both', expand=True)
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)
    panel = ttk.Frame(canvas, padding=12)
    window = canvas.create_window((0, 0), window=panel, anchor='nw')
    panel.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
    root.bind('<MouseWheel>', lambda event: canvas.yview_scroll(int(-event.delta / 120), 'units'))
    panel.columnconfigure(0, weight=1)

    # Preserve JSON types for carrier/chord choices and enable/freeze switches.
    variables = {}
    for key, value in DEFAULT_SETTINGS.items():
        kind = tk.BooleanVar if isinstance(value, bool) else tk.StringVar if isinstance(value, str) else tk.DoubleVar
        variables[key] = kind(value=value)
    selected = {kind: tk.StringVar(value='Custom') for kind in ('preset', 'voice', 'speaker')}
    applying = False
    settings = {}
    repeat_trigger = 0
    diagnostics = {'dropouts': 0, 'error': ''}
    menus = {}
    row = 0

    def snapshot():
        # Reading Tk values belongs exclusively to the GUI thread.
        values = {key: variable.get() for key, variable in variables.items()}
        values['bit_depth'] = int(round(values['bit_depth']))
        values['sample_rate'] = int(round(values['sample_rate']))
        values['vocoder_bands'] = int(round(values['vocoder_bands']))
        return values

    def apply_values(values):
        nonlocal applying
        applying = True
        try:
            for key, value in values.items():
                if key in variables:
                    variables[key].set(value)
        finally:
            applying = False

    def apply_preset(name):
        # A complete snapshot prevents settings leaking from one preset to another.
        values = complete_preset(presets[name], voices, speakers)
        apply_values(values)
        selected['preset'].set(name)
        selected['voice'].set(values.get('voice_type', 'Custom'))
        selected['speaker'].set(values.get('speaker_type', 'Custom'))
        # A preset can customize a base profile; don't label those controls as an
        # exact profile match when its stored snapshot has different values.
        for kind, profiles, keys in [('voice', voices, VOICE_KEYS), ('speaker', speakers, SPEAKER_KEYS)]:
            profile = profiles.get(selected[kind].get(), {})
            if any(values[key] != profile.get(key, DEFAULT_SETTINGS[key]) for key in keys):
                selected[kind].set('Custom')

    def apply_profile(kind, name):
        keys = VOICE_KEYS if kind == 'voice' else SPEAKER_KEYS
        profile = (voices if kind == 'voice' else speakers)[name]
        apply_values({key: profile.get(key, DEFAULT_SETTINGS[key]) for key in keys})
        selected[kind].set(name)
        selected['preset'].set('Custom')

    def changed(key):
        if applying:
            return
        selected['preset'].set('Custom')
        if key in VOICE_KEYS:
            selected['voice'].set('Custom')
        if key in SPEAKER_KEYS:
            selected['speaker'].set('Custom')

    previous_random = None
    def randomize():
        nonlocal previous_random
        previous_random = (snapshot(), {key: value.get() for key, value in selected.items()})
        apply_values(randomize_settings(snapshot()))
        for value in selected.values():
            value.set('Custom')
        undo_button.configure(state='normal')

    def undo_randomize():
        nonlocal previous_random
        if previous_random is not None:
            values, names = previous_random
            apply_values(values)
            for key, value in names.items():
                selected[key].set(value)
            previous_random = None
            undo_button.configure(state='disabled')

    # Keep experimentation and one-step recovery beside the always-visible gains.
    random_controls = ttk.Frame(root, padding=(12, 5))
    random_controls.pack(fill='x', before=container)
    ttk.Button(random_controls, text='Randomize settings', command=randomize).pack(side='left')
    undo_button = ttk.Button(random_controls, text='Undo randomize', command=undo_randomize, state='disabled')
    undo_button.pack(side='left', padx=6)

    for key, variable in variables.items():
        variable.trace_add('write', lambda *args, key=key: changed(key))

    def rebuild(kind, values, callback):
        menu = menus[kind]['menu']
        menu.delete(0, 'end')
        categories = {}
        for name in values:
            # Group the expansion library so a large preset collection remains
            # navigable. The saved/selected name still includes its category.
            target, label = menu, name
            if kind == 'preset' and ' / ' in name:
                category, label = name.split(' / ', 1)
                if category not in categories:
                    categories[category] = tk.Menu(menu, tearoff=False)
                    menu.add_cascade(label=category, menu=categories[category])
                target = categories[category]
            target.add_command(label=label, command=lambda name=name: callback(name))

    def profile_controls(kind, title, values, filename, callback):
        nonlocal row
        ttk.Label(panel, text=title, font=('Arial', 10, 'bold')).grid(row=row, sticky='w', pady=(12, 3))
        row += 1
        menu = ttk.OptionMenu(panel, selected[kind], selected[kind].get())
        menu.grid(row=row, sticky='ew')
        menus[kind] = menu
        rebuild(kind, values, callback)
        row += 1
        box = ttk.Frame(panel)
        box.grid(row=row, sticky='ew', pady=4)
        box.columnconfigure(0, weight=1)
        name_entry = ttk.Entry(box)
        name_entry.grid(row=0, column=0, sticky='ew')

        def save():
            name = name_entry.get().strip()
            if not name:
                return
            current = snapshot()
            if kind == 'preset':
                current.update(voice_type=selected['voice'].get(), speaker_type=selected['speaker'].get())
            else:
                keys = VOICE_KEYS if kind == 'voice' else SPEAKER_KEYS
                current = {key: current[key] for key in keys}
            values[name] = current
            save_profiles(filename, values)
            # Rebuild rather than append, so overwriting does not duplicate menu items.
            rebuild(kind, values, callback)
            selected[kind].set(name)

        ttk.Button(box, text='Save as', command=save).grid(row=0, column=1, padx=4)
        if kind == 'preset':
            def delete():
                name = selected[kind].get()
                if name == 'Default' or name not in values:
                    return
                del values[name]
                save_profiles(filename, values)
                rebuild(kind, values, callback)
                if values:
                    callback('Default' if 'Default' in values else next(iter(values)))
            ttk.Button(box, text='Delete', command=delete).grid(row=0, column=2)
        row += 1

    profile_controls('preset', 'Preset — restores all controls', presets, 'voice_presets.json', apply_preset)
    profile_controls('voice', 'Voice type — metallic character', voices, 'voice_types.json', lambda name: apply_profile('voice', name))
    profile_controls('speaker', 'Speaker type — compression and digital quality', speakers, 'speaker_types.json', lambda name: apply_profile('speaker', name))
    ttk.Label(panel, text='Damage mode').grid(row=row, sticky='w', pady=(12, 0))
    row += 1
    ttk.Combobox(panel, textvariable=variables['mode'], state='readonly',
                 values=('normal', 'slightly_broken', 'extremely_damaged')).grid(row=row, sticky='ew')
    row += 1
    def repeat_now():
        nonlocal repeat_trigger
        repeat_trigger += 1
    ttk.Button(panel, text='Repeat syllable now (while speaking)', command=repeat_now).grid(row=row, sticky='ew', pady=4)
    row += 1
    ttk.Checkbutton(panel, text='Mix at least 50% original voice', variable=variables['hear_original']).grid(row=row, sticky='w', pady=6)
    row += 1

    ttk.Label(panel, text='Vocoder — speech-shaped synthesizer', font=('Arial', 10, 'bold')).grid(row=row, sticky='w', pady=(12, 3))
    row += 1
    ttk.Checkbutton(panel, text='Enable vocoder', variable=variables['vocoder_enabled']).grid(row=row, sticky='w')
    row += 1
    for key, label, choices in [('vocoder_carrier', 'Carrier waveform', CARRIERS),
                                ('vocoder_chord', 'Carrier chord', tuple(CHORDS))]:
        ttk.Label(panel, text=label).grid(row=row, sticky='w')
        row += 1
        ttk.Combobox(panel, textvariable=variables[key], state='readonly', values=choices).grid(row=row, sticky='ew')
        row += 1
    ttk.Checkbutton(panel, text='Freeze vowel / spectral envelope (holds a tone)',
                    variable=variables['vocoder_freeze']).grid(row=row, sticky='w')
    row += 1

    controls = [
        ('vocoder_mix', 'Vocoder wet mix', 0, 1),
        ('vocoder_note', 'Carrier note (MIDI; 45 = A2, 60 = C4)', 24, 84),
        ('vocoder_detune', 'Carrier detune (cents)', 0, 50),
        ('vocoder_pulse_width', 'Pulse width (pulse carrier only)', .1, .9),
        ('vocoder_noise', 'Noise blended into pitched carrier', 0, 1),
        ('vocoder_bands', 'Vocoder bands (detail)', 8, 32),
        ('vocoder_bandwidth', 'Band width', .6, 1.8),
        ('vocoder_attack_ms', 'Envelope attack (ms)', 1, 100),
        ('vocoder_release_ms', 'Envelope release (ms)', 5, 500),
        ('vocoder_formant', 'Formant shift (semitones)', -12, 12),
        ('vocoder_consonants', 'Consonant preservation', 0, 1),
        ('vocoder_gain_db', 'Vocoder gain (dB)', -18, 18),
        ('robot_mix', 'Robot / effect mix', 0, 1),
        ('pitch_shift', 'Pitch (semitones)', -12, 12),
        ('modulation_rate', 'Metallic carrier (Hz)', 20, 200),
        ('modulation_depth', 'Metallic modulation depth', 0, 1),
        ('bandpass_freq', 'Filter color (Hz; 0 bypasses)', 0, 2000),
        ('distortion_amount', 'Distortion', 0, 100),
        ('syllable_ms', 'Repeated syllable length (ms)', 180, 350),
        ('repeat_interval', 'Speech time between repeats (seconds)', .5, 5),
        ('repeat_strength', 'Syllable repeat strength', 0, 1),
        ('repeat_threshold_db', 'Repeat speech detection threshold (dB)', -80, -20),
        ('sample_rate', 'Speaker effective sample rate (Hz)', 4000, 44100),
        ('compression_threshold_db', 'Compression threshold (dB)', -60, 0),
        ('compression_ratio', 'Compression ratio (1 bypasses)', 1, 20),
        ('makeup_db', 'Compression makeup gain (dB)', 0, 18),
        ('bit_depth', 'Bitcrush depth (16 bypasses)', 4, 16),
        ('bitcrush_mix', 'Bitcrush wet mix (0 bypasses)', 0, 1),
        ('bitcrush_drive_db', 'Bitcrush input drive (dB)', 0, 24),
        ('crackle', 'Speaker crackle texture', 0, .3),
        ('gain', 'Preset gain (saved with this voice)', .1, 3),
    ]
    for key, label, low, high in controls:
        if key == 'robot_mix':
            ttk.Separator(panel).grid(row=row, sticky='ew', pady=12)
            row += 1
            ttk.Label(panel, text='Voice, damage, and speaker controls', font=('Arial', 10, 'bold')).grid(row=row, sticky='w')
            row += 1
        line = ttk.Frame(panel)
        line.grid(row=row, sticky='ew', pady=(7, 0))
        ttk.Label(line, text=label).pack(side='left')
        readout = ttk.Label(line)
        readout.pack(side='right')
        def update_readout(*args, key=key, readout=readout):
            readout.config(text=f'{variables[key].get():.2f}')
        variables[key].trace_add('write', update_readout)
        update_readout()
        row += 1
        ttk.Scale(panel, from_=low, to=high, variable=variables[key]).grid(row=row, sticky='ew')
        row += 1

    # Actually apply the displayed startup preset after every widget exists.
    apply_preset('Default' if 'Default' in presets else next(iter(presets)))
    def sync_settings():
        nonlocal settings
        settings = dict(snapshot(), enabled=enabled.get(), repeat_trigger=repeat_trigger,
                        output_volume=master_var.get() / 100, playback_volume=playback_var.get() / 100,
                        mic_gain=mic_var.get() / 100)
        state = 'Enabled' if enabled.get() else 'Disabled — microphone passthrough'
        status_text.set(f"{state} | Repeats: {diagnostics.get('repeats', 0)} | Audio dropouts: {diagnostics['dropouts']}"
                        f" | Compression: {diagnostics.get('compression_db', 0):.1f} dB reduction" +
                        (f" | {diagnostics['error']}" if diagnostics['error'] else ''))
        root.after(30, sync_settings)
    sync_settings()

    # Warm up library loading before microphone callbacks begin.
    follow_envelope(np.zeros(8), 0., .9, .99)
    VoiceProcessor().process(np.zeros(CHUNK, dtype=np.float32), dict(enabled=True, pitch_shift=-3))
    processor = VoiceProcessor()
    def audio_callback(indata, outdata, frames, time, status):
        if status:
            diagnostics['dropouts'] += 1
        try:
            current = settings  # Use one consistent GUI snapshot for both routes.
            processed = processor.process(indata[:, 0], current)
            diagnostics['repeats'] = processor.repeat_events
            diagnostics['compression_db'] = processor.compression_reduction_db
            outdata[:, 0] = playback_audio(processed, current['playback_volume'], monitor)
            if output_queue is not None:
                publish_audio(output_queue, processed)
        except Exception as error:
            # Surface callback failures in the GUI rather than silently stopping audio.
            outdata.fill(0)
            diagnostics['error'] = str(error)

    try:
        with sd.Stream(samplerate=SAMPLERATE, blocksize=CHUNK, channels=1,
                       dtype='float32', callback=audio_callback):
            root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == '__main__':
    run_gui()
