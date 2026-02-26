import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.signal import butter, lfilter
import noisereduce as nr
import librosa
import json
import os

SAMPLERATE = 44100 # standard CD-quality sample rate
CHUNK = 8192  # number of samples per audio block (20ms at 44.1kHz)

PRESET_FILE = 'voice_presets.json'
VOICE_TYPE_FILE = 'voice_types.json'

# Voice changer state controller
class VoiceChangerController:
    def __init__(self):
        self.enabled = False
        self.mode = 'normal'
        self.hear_original = True
        self.pitch_shift = 0.0     # semitones
        self.distortion_amount = 0 # 0 to 100

    def set_enabled(self, val):
        self.enabled = val

    def set_mode(self, mode):
        self.mode = mode

    def set_hear_original(self, val):
        self.hear_original = val

    def set_pitch(self, val):
        self.pitch_shift = val

    def set_distortion(self, val):
        self.distortion_amount = val

def pitch_shift(data, shift_semitones):
    """
    Use librosa to shift pitch without changing speed.
    Data must be float32 numpy array!
    """
    # Librosa expects mono float32 numpy arrays
    data = data.astype(np.float32)
    # Pitch shift. Librosa expects sample rate.
    shifted = librosa.effects.pitch_shift(data, sr=SAMPLERATE, n_steps=shift_semitones)
    # Librosa may return a slightly different length, so crop/pad as needed
    if len(shifted) < len(data):
        shifted = np.pad(shifted, (0, len(data)-len(shifted)), mode='constant')
    elif len(shifted) > len(data):
        shifted = shifted[:len(data)]
    return shifted

def distortion(data, amount):
    """Basic waveshaping distortion. Amount from 0-100."""
    if amount == 0:
        return data
    amount = min(max(amount, 0), 100)
    drive = amount / 100 * 15  # adjust factor for taste
    # tanh distortion
    distorted = np.tanh(data * drive)
    return distorted

def bit_crush(data, bit_depth):
    """Reduce bit depth to create lo-fi digital effect. bit_depth: 4-16 bits."""
    if bit_depth >= 16:  # No effect at CD quality
        return data
    
    # Quantize to fewer bits
    levels = 2 ** bit_depth
    quantized = np.round(data * (levels - 1)) / (levels - 1)
    return quantized



class RobotBuffer:
    def __init__(self, buffer_size):
        self.buffer = np.zeros(buffer_size, dtype=np.float32)
        self.size = buffer_size
        self.write_idx = 0
        self.full = False
        self.replay_start = None
        self.replay_len = 0
        self.replay_count = 0
        self.replays_left = 0
        self.playing_replay = False

    def push(self, data):
        n = len(data)
        if n > self.size:
            data = data[-self.size:]  # Only keep the last portion if data is too long
            n = len(data)
        # Write to circular buffer
        end_idx = (self.write_idx + n) % self.size
        if end_idx > self.write_idx:
            self.buffer[self.write_idx:end_idx] = data
        else:
            chunk = self.size - self.write_idx
            self.buffer[self.write_idx:] = data[:chunk]
            self.buffer[:end_idx] = data[chunk:]
        self.write_idx = end_idx
        if self.write_idx == 0:
            self.full = True

    def start_replay(self, chunk_len, repeats):
        # Only start if the buffer is full or we've written enough
        if self.write_idx >= chunk_len or self.full:
            idx = (self.write_idx - chunk_len) % self.size
            self.replay_start = idx
            self.replay_len = chunk_len
            self.replay_count = repeats
            self.replays_left = repeats
            self.playing_replay = True

    def get_segment(self, n):
        # Provide n samples, either from buffer (normal) or from replay region
        if not self.playing_replay or self.replays_left == 0:
            # Normal operation: return latest n samples from buffer end
            start = (self.write_idx - n) % self.size
            if start + n <= self.size:
                return self.buffer[start:start+n].copy()
            else:
                chunk = self.size - start
                return np.concatenate((self.buffer[start:], self.buffer[:n-chunk]))
        else:
            # Replay: provide from the stored segment
            seg = []
            while len(seg) < n and self.replays_left > 0:
                seg_needed = min(n - len(seg), self.replay_len)
                begin = (self.replay_start) % self.size
                end = (self.replay_start + seg_needed) % self.size
                if begin + seg_needed <= self.size:
                    seg.append(self.buffer[begin:begin+seg_needed])
                else:
                    first = self.size - begin
                    seg.append(self.buffer[begin:])
                    seg.append(self.buffer[:seg_needed - first])
                self.replay_start = (self.replay_start + seg_needed) % self.size
                # End of one repeat
                if ((self.replay_start - (self.write_idx - self.replay_len)) % self.size) == 0:
                    self.replays_left -= 1
                    self.replay_start = (self.write_idx - self.replay_len) % self.size
                    if self.replays_left == 0:
                        self.playing_replay = False
            return np.concatenate(seg)[:n]

robot_buffer = RobotBuffer(SAMPLERATE) # 1 second buffer for robot effect

def robot_effect_live(indata, mode="normal"):
    # Feed the circular buffer with indata
    robot_buffer.push(indata)
    # Decide if we want to stutter
    if mode in ("slightly_broken", "extremely_damaged"):
        if not robot_buffer.playing_replay and np.random.rand() < (0.05 if mode=="slightly_broken" else 0.15):
            # Random "syllable" length between 0.2 and 0.5 seconds
            min_len = int(0.2 * SAMPLERATE)
            max_len = int(0.5 * SAMPLERATE)
            seg_len = np.random.randint(min_len, max_len + 1)
            repeats = np.random.randint(2, 10) if mode=="slightly_broken" else np.random.randint(6, 15)
            robot_buffer.start_replay(seg_len, repeats)
        # Output either the latest, or (if playing replay) a repeated chunk:
        return robot_buffer.get_segment(len(indata))
    else:
        return indata

# GUI setup
def run_gui():
    root = tk.Tk()
    root.title("Voice Changer")
    root.geometry("400x900")  # Set window size

    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)

    # ==== ENABLE/DISABLE ====
    enabled_var = tk.BooleanVar(value=False)
    def toggle_enabled():
        controller.set_enabled(enabled_var.get())
    enable_check = ttk.Checkbutton(main_frame, text="Enable Voice Changer", variable=enabled_var, command=toggle_enabled)
    enable_check.grid(row=0, column=0, sticky='w', pady=(0, 10))

    status_label = ttk.Label(main_frame, text="Voice changer disabled")
    status_label.grid(row=1, column=0, sticky='w', pady=(0, 15))
    def update_status_label():
        status = f"Voice changer {'enabled' if controller.enabled else 'disabled'}"
        if controller.enabled:
            status += f" ({controller.mode}, Pitch: {controller.pitch_shift:+.1f}, Distortion: {controller.distortion_amount})"
        status_label.config(text=status)
        root.after(200, update_status_label)
    root.after(200, update_status_label)

    # ==== EFFECT MODE ====
    ttk.Label(main_frame, text="Effect Mode:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky='w', pady=(10, 5))
    mode_var = tk.StringVar(value="normal")
    def set_mode():
        controller.set_mode(mode_var.get())
    modes = [("Normal", "normal"), ("Slightly Broken", "slightly_broken"), ("Extremely Damaged", "extremely_damaged")]
    for idx, (label, value) in enumerate(modes):
        rb = ttk.Radiobutton(main_frame, text=label, variable=mode_var, value=value, command=set_mode)
        rb.grid(row=3+idx, column=0, sticky='w', padx=20)

    hear_original_var = tk.BooleanVar(value=True)
    def toggle_hear_original():
        controller.set_hear_original(hear_original_var.get())
    hear_orig_check = ttk.Checkbutton(main_frame, text="Hear Original Voice", variable=hear_original_var, command=toggle_hear_original)
    hear_orig_check.grid(row=6, column=0, sticky='w', padx=20, pady=(5, 15))

    # ==== PRESET MANAGEMENT ====
    ttk.Label(main_frame, text="Presets:", font=("Arial", 10, "bold")).grid(row=7, column=0, sticky='w', pady=(10, 5))
    presets = load_presets()
    preset_var = tk.StringVar(value=list(presets.keys())[0])

    def apply_preset(name):
        params = presets[name]
        controller.set_mode(params["mode"])
        controller.set_pitch(params["pitch_shift"])
        controller.set_distortion(params["distortion_amount"])
        controller.set_hear_original(params["hear_original"])
        mode_var.set(params["mode"])
        pitch_var.set(params["pitch_shift"])
        dist_var.set(params["distortion_amount"])
        hear_original_var.set(params["hear_original"])

    preset_menu = ttk.OptionMenu(main_frame, preset_var, preset_var.get(), *presets.keys(), command=apply_preset)
    preset_menu.grid(row=8, column=0, sticky='ew', pady=(0, 5))

    preset_entry = ttk.Entry(main_frame)
    preset_entry.grid(row=9, column=0, sticky='ew', pady=(0, 5))

    def save_current_preset():
        name = preset_entry.get().strip()
        if not name:
            return
        presets[name] = {
            "mode": controller.mode,
            "pitch_shift": controller.pitch_shift,
            "distortion_amount": controller.distortion_amount,
            "hear_original": controller.hear_original
        }
        save_presets(presets)
        preset_menu['menu'].add_command(label=name, command=tk._setit(preset_var, name, apply_preset))

    def delete_preset():
        name = preset_var.get()
        if name == "Default":
            return
        presets.pop(name, None)
        save_presets(presets)
        preset_menu['menu'].delete(0, 'end')
        for preset_name in presets.keys():
            preset_menu['menu'].add_command(
                label=preset_name,
                command=tk._setit(preset_var, preset_name, apply_preset)
            )
        preset_var.set("Default")
        apply_preset("Default")

    save_button = ttk.Button(main_frame, text="Save Current As Preset", command=save_current_preset)
    save_button.grid(row=10, column=0, sticky='ew', pady=(0, 5))

    delete_button = ttk.Button(main_frame, text="Delete Preset", command=delete_preset)
    delete_button.grid(row=11, column=0, sticky='ew', pady=(0, 15))

    # ==== VOICE TYPE MANAGEMENT ====
    voice_types = load_voice_types()
    voice_type_var = tk.StringVar(value=list(voice_types.keys())[0])

    ttk.Label(main_frame, text="Voice Types:", font=("Arial", 10, "bold")).grid(row=12, column=0, sticky='w', pady=(10, 5))

    def apply_voice_type(name):
        params = voice_types[name]
        pitch_var.set(params.get("pitch_shift", 0))
        bandpass_var.set(params.get("bandpass_freq", 600))
        mod_rate_var.set(params.get("modulation_rate", 0))
        mod_depth_var.set(params.get("modulation_depth", 0))

    voice_type_menu = ttk.OptionMenu(main_frame, voice_type_var, voice_type_var.get(), *voice_types.keys(), command=apply_voice_type)
    voice_type_menu.grid(row=13, column=0, sticky='ew', pady=(0, 5))

    voice_type_entry = ttk.Entry(main_frame)
    voice_type_entry.grid(row=14, column=0, sticky='ew', pady=(0, 5))

    def save_current_voice_type():
        name = voice_type_entry.get().strip()
        if not name:
            return
        voice_types[name] = {
            "pitch_shift": pitch_var.get(),
            "bandpass_freq": bandpass_var.get(),
            "modulation_rate": mod_rate_var.get(),
            "modulation_depth": mod_depth_var.get()
        }
        save_voice_types(voice_types)

    save_voice_type_button = ttk.Button(main_frame, text="Save Current As Voice Type", command=save_current_voice_type)
    save_voice_type_button.grid(row=15, column=0, sticky='ew', pady=(0, 15))

    # ==== EFFECT CONTROLS ====
    ttk.Label(main_frame, text="Effect Controls:", font=("Arial", 10, "bold")).grid(row=16, column=0, sticky='w', pady=(10, 5))

    pitch_var = tk.DoubleVar(value=0.0)
    def set_pitch(val):
        controller.set_pitch(float(val))
    ttk.Label(main_frame, text="Pitch Shift (semitones):").grid(row=17, column=0, sticky='w')
    pitch_slider = ttk.Scale(main_frame, from_=-12, to=12, variable=pitch_var, command=set_pitch)
    pitch_slider.grid(row=18, column=0, sticky='ew', pady=(0, 10))

    dist_var = tk.IntVar(value=0)
    def set_distortion(val):
        controller.set_distortion(int(float(val)))
    ttk.Label(main_frame, text="Distortion (0-100):").grid(row=19, column=0, sticky='w')
    dist_slider = ttk.Scale(main_frame, from_=0, to=100, variable=dist_var, command=set_distortion)
    dist_slider.grid(row=20, column=0, sticky='ew', pady=(0, 10))

    bandpass_var = tk.IntVar(value=600)
    ttk.Label(main_frame, text="Bandpass Frequency (Hz):").grid(row=21, column=0, sticky='w')
    bandpass_slider = ttk.Scale(main_frame, from_=200, to=2000, variable=bandpass_var)
    bandpass_slider.grid(row=22, column=0, sticky='ew', pady=(0, 10))

    mod_rate_var = tk.DoubleVar(value=0.0)
    ttk.Label(main_frame, text="Modulation Rate (Hz):").grid(row=23, column=0, sticky='w')
    mod_rate_slider = ttk.Scale(main_frame, from_=0, to=20, variable=mod_rate_var)
    mod_rate_slider.grid(row=24, column=0, sticky='ew', pady=(0, 10))

    mod_depth_var = tk.DoubleVar(value=0.0)
    ttk.Label(main_frame, text="Modulation Depth (0–1):").grid(row=25, column=0, sticky='w')
    mod_depth_slider = ttk.Scale(main_frame, from_=0, to=1, variable=mod_depth_var)
    mod_depth_slider.grid(row=26, column=0, sticky='ew', pady=(0, 10))

    bitcrush_var = tk.IntVar(value=16)
    ttk.Label(main_frame, text="Bit Depth (4-16):").grid(row=27, column=0, sticky='w')
    bitcrush_slider = ttk.Scale(main_frame, from_=4, to=16, variable=bitcrush_var, orient='horizontal')
    bitcrush_slider.grid(row=28, column=0, sticky='ew', pady=(0, 10))

    gain_var = tk.DoubleVar(value=1.0)
    ttk.Label(main_frame, text="Output Gain (0.1-3.0):").grid(row=29, column=0, sticky='w')
    gain_slider = ttk.Scale(main_frame, from_=0.1, to=3.0, variable=gain_var, orient='horizontal')
    gain_slider.grid(row=30, column=0, sticky='ew', pady=(0, 15))

    #audio processing callback
    def apply_voice_type_effects(data, pitch_value, bandpass_freq, mod_rate, mod_depth):
        
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) # Ensure no NaN or Inf values before processing

        # Pitch shift (already handled by pitch_shift function)
        #data = pitch_shift(data, pitch_value) #called elsewhere so no need
        
        # Bandpass filtering (using scipy)
        if bandpass_freq > 0:
            nyq = 0.5 * SAMPLERATE
            window_width = 400 # Hz window around the center frequency
            low = max(100, bandpass_freq - window_width) / nyq
            high = min(SAMPLERATE // 2, bandpass_freq + window_width) / nyq
            b, a = butter(2, [low, high], btype='band')
            data = lfilter(b, a, data)
        
        # Modulation (robotic vibrato/ring modulation)
        if mod_rate > 0 and mod_depth > 0:
            t = np.arange(len(data)) / SAMPLERATE
            mod_wave = 1 + mod_depth * np.sin(2 * np.pi * mod_rate * t)
            data = data * mod_wave

        return data

    def noise_gate_soft(data, threshold=0.02, fade=0.01):
        # Linearly fade signals that are just above threshold
        gate = np.abs(data)
        mask = gate > threshold
        fade_mask = (gate > threshold - fade) & (gate <= threshold)
        result = np.zeros_like(data)
        result[mask] = data[mask]
        # Apply fade transition for values just above threshold
        result[fade_mask] = data[fade_mask] * ((gate[fade_mask] - (threshold - fade)) / fade)
        return result

    def denoise(data):
        # Assumes data is a mono numpy array of float32
        reduced_noise = nr.reduce_noise(y=data, sr=SAMPLERATE, prop_decrease=0.7)
        return reduced_noise

    def audio_callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        mono = indata[:, 0]

        #denoise before processing to help with cleaner effects, especially for distortion
        mono = noise_gate_soft(mono, threshold=0.02, fade=0.01)
        # Replace NaN and Inf with 0 before passing to pitch_shift
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

        mono = denoise(mono)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

        monoGAIN = 1.5  # Adjust as needed or use a slider in your GUI

        # After all effects in your processing pipeline:
        mono = mono * monoGAIN

        # Limiter/normalization step to avoid exceeding maximum amplitude:
        max_allowed = 0.8  # Prevents the output from being too loud (adjust as needed)
        if np.max(np.abs(mono)) > max_allowed:
            mono = mono / np.max(np.abs(mono)) * max_allowed

        # Optionally, use soft clipping for extra protection:
        mono = np.tanh(mono)


        # Get live values from sliders/widgets
        vtype = voice_type_var.get()
        vparams = voice_types[vtype]
        pitch = pitch_var.get()
        band = bandpass_var.get()
        modrate = mod_rate_var.get()
        moddepth = mod_depth_var.get()
        # Apply voice type effects
        processed = apply_voice_type_effects(
            mono,
            pitch,
            band,
            modrate,
            moddepth
        )

        #denoise to help with cleaner effects
        processed = noise_gate_soft(processed, threshold=0.02, fade=0.01)
        # Replace NaN and Inf with 0 before passing to pitch_shift
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

        processed = denoise(processed)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

        #apply pitch shift, distortion, and bitcrush based on sliders
        if controller.pitch_shift != 0.0:
            processed = pitch_shift(processed, controller.pitch_shift)
        if controller.distortion_amount > 0:
            processed = distortion(processed, controller.distortion_amount)
        if bitcrush_var.get() < 16:
            processed = bit_crush(processed, bitcrush_var.get())

        #denoise to help with cleaner effects
        processed = noise_gate_soft(processed, threshold=0.02, fade=0.01)
        # Replace NaN and Inf with 0 before passing to pitch_shift
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

        processed = denoise(processed)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply robot mode
        processed = robot_effect_live(
            processed,
            controller.mode
        )

        #denoise to help with cleaner effects
        processed = noise_gate_soft(processed, threshold=0.02, fade=0.01)
        # Replace NaN and Inf with 0 before passing to pitch_shift
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

        processed = denoise(processed)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

        GAIN = gain_var.get()  # Adjust as needed or use a slider in your GUI

        # After all effects in your processing pipeline:
        processed = processed * GAIN

        # Limiter/normalization step to avoid exceeding maximum amplitude:
        max_allowed = 0.8  # Prevents the output from being too loud (adjust as needed)
        if np.max(np.abs(processed)) > max_allowed:
            processed = processed / np.max(np.abs(processed)) * max_allowed

        # Optionally, use soft clipping for extra protection:
        processed = np.tanh(processed)

        if controller.hear_original:
            processed = (processed + mono) / 2
        
        outdata[:, 0] = processed

    # Start audio stream
    stream = sd.Stream(samplerate=SAMPLERATE, blocksize=CHUNK, channels=1, dtype='float32', callback=audio_callback)
    stream.start()
    root.protocol("WM_DELETE_WINDOW", lambda: (stream.stop(), root.destroy()))
    root.mainloop()

#voice preset management
def load_presets():
    # Load from JSON file; if not present or empty, use defaults
    if os.path.exists(PRESET_FILE):
        with open(PRESET_FILE, 'r') as f:
            presets = json.load(f)
        if not presets:  # file exists but is empty
            presets = {
                "Default": {
                    "mode": "normal",
                    "pitch_shift": 0.0,
                    "distortion_amount": 0,
                    "hear_original": True
                },
                "Glitchy Robot": {
                    "mode": "extremely_damaged",
                    "pitch_shift": -5,
                    "distortion_amount": 60,
                    "hear_original": False
                },
                "Cyber Warforged": {
                    "mode": "slightly_broken",
                    "pitch_shift": +8,
                    "distortion_amount": 25,
                    "hear_original": True
                },
                # Add these new presets:
                "G1 Soundwave": {
                    "mode": "normal",
                    "pitch_shift": -12,          # Lower pitch for deep robotic voice
                    "distortion_amount": 55,     # Moderate distortion for metallic effect
                    "hear_original": False
                },
                "HK-74": {
                    "mode": "slightly_broken",   # Some glitch for terminator-like effect
                    "pitch_shift": -7,
                    "distortion_amount": 35,
                    "hear_original": False
                },
                "AUTO (Wall-E)": {
                    "mode": "normal",            # Very clean, monotone robotic
                    "pitch_shift": -2,
                    "distortion_amount": 10,     # Just a touch of processing
                    "hear_original": True
                }
            }
            save_presets(presets)  # Save defaults to file
        return presets
    else:
        # Initial default presets
        presets = {
                "Default": {
                    "mode": "normal",
                    "pitch_shift": 0.0,
                    "distortion_amount": 0,
                    "hear_original": True
                },
                "Glitchy Robot": {
                    "mode": "extremely_damaged",
                    "pitch_shift": -5,
                    "distortion_amount": 60,
                    "hear_original": False
                },
                "Cyber Warforged": {
                    "mode": "slightly_broken",
                    "pitch_shift": +8,
                    "distortion_amount": 25,
                    "hear_original": True
                },
                # Add these new presets:
                "G1 Soundwave": {
                    "mode": "normal",
                    "pitch_shift": -12,          # Lower pitch for deep robotic voice
                    "distortion_amount": 55,     # Moderate distortion for metallic effect
                    "hear_original": False
                },
                "HK-74": {
                    "mode": "slightly_broken",   # Some glitch for terminator-like effect
                    "pitch_shift": -7,
                    "distortion_amount": 35,
                    "hear_original": False
                },
                "AUTO (Wall-E)": {
                    "mode": "normal",            # Very clean, monotone robotic
                    "pitch_shift": -2,
                    "distortion_amount": 10,     # Just a touch of processing
                    "hear_original": True
                }
            }
        save_presets(presets)
        return presets

#voice type management
def load_voice_types():
    if os.path.exists(VOICE_TYPE_FILE):
        with open(VOICE_TYPE_FILE, 'r') as f:
            return json.load(f)
    else:
        voice_types = {
                "Default": {
                "pitch_shift": 0,
                "bandpass_freq": 600,
                "modulation_rate": 0,
                "modulation_depth": 0
                },
                "G1 Soundwave": {
                "pitch_shift": -12,
                "bandpass_freq": 850,
                "modulation_rate": 8,
                "modulation_depth": 0.55
                },
                "HK-74": {
                "pitch_shift": -7,
                "bandpass_freq": 700,
                "modulation_rate": 3,
                "modulation_depth": 0.35
                },
                "AUTO (Wall-E)": {
                "pitch_shift": -2,
                "bandpass_freq": 1100,
                "modulation_rate": 1.5,
                "modulation_depth": 0.20
                }
                }
        save_voice_types(voice_types)
        return voice_types

def save_voice_types(voice_types):
    with open(VOICE_TYPE_FILE, 'w') as f:
        json.dump(voice_types, f, indent=2)

def save_presets(presets):
    with open(PRESET_FILE, 'w') as f:
        json.dump(presets, f, indent=2)

if __name__ == "__main__":
    controller = VoiceChangerController()
    run_gui()