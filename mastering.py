import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from scipy.signal import butter, lfilter
import logging
import threading
from PIL import Image, ImageTk
import os
import soundfile as sf
import queue
import sys
import time
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pyaudio

# ==== Logging Setup ====
logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG to see all logs
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("audio_mastering.log"),
                        logging.StreamHandler(sys.stdout)  # Also log to console
                    ])

# ==== GUI Setup ====
root = tk.Tk()
root.title("Audio Mastering Tool")
root.geometry("1200x800")
root.configure(bg='#f0f0f0')  # Light background for a modern look

# Apply a custom style to ttk widgets
style = ttk.Style(root)
style.theme_use('clam')

# Define custom colors for a modern look
primary_color = '#0078D7'  # Modern blue
secondary_color = '#005A9E'  # Darker blue
background_color = '#f0f0f0'  # Light background
foreground_color = '#000000'  # Black text

# Configure styles
style.configure('TFrame', background=background_color)
style.configure('TLabel', background=background_color, foreground=foreground_color, font=('Helvetica', 12))
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TCheckbutton', background=background_color, foreground=foreground_color, font=('Helvetica', 10))
style.configure('Horizontal.TScale', background=background_color)

# Create custom styles for buttons
style.configure('Normal.TButton',
                background=primary_color,
                foreground='white',
                borderwidth=1,
                focusthickness=3,
                focuscolor='none')
style.map('Normal.TButton',
          background=[('active', secondary_color), ('disabled', '#d9d9d9')],
          foreground=[('disabled', '#a3a3a3')])

# Load icons for media player buttons
try:
    play_icon_img = Image.open(os.path.join("ui", "play.png")).resize((30, 30))
    play_icon = ImageTk.PhotoImage(play_icon_img)

    pause_icon_img = Image.open(os.path.join("ui", "pause.png")).resize((30, 30))
    pause_icon = ImageTk.PhotoImage(pause_icon_img)

    stop_icon_img = Image.open(os.path.join("ui", "stop.png")).resize((30, 30))
    stop_icon = ImageTk.PhotoImage(stop_icon_img)
except Exception as e:
    logging.error("Error loading media player icons: %s", e)
    play_icon = pause_icon = stop_icon = None

# Variables for audio path and adjustments
audio_path = None
audio_data = None
sr = None
update_queue = queue.Queue()

# Playback variables
is_playing = False
playback_thread = None
playback_position = 0  # in samples
playback_lock = threading.Lock()
pyaudio_instance = None
stream = None
volume = tk.DoubleVar(value=0.5)
elapsed_time = tk.StringVar(value="00:00")
total_duration = tk.StringVar(value="00:00")
duration = 0  # Total duration in seconds

# Mastering settings
noise_reduction_strength = tk.DoubleVar(value=1.0)
low_gain = tk.DoubleVar(value=1.0)
mid_gain = tk.DoubleVar(value=1.0)
high_gain = tk.DoubleVar(value=1.0)
compression_threshold = tk.DoubleVar(value=-20)
compression_ratio = tk.DoubleVar(value=4.0)
target_loudness = tk.DoubleVar(value=-14)

# Variables for selective mastering options
apply_noise_reduction = tk.IntVar(value=1)
apply_eq = tk.IntVar(value=1)
apply_compression_var = tk.IntVar(value=1)
apply_loudness_norm = tk.IntVar(value=1)

# Progress bar
progress = ttk.Progressbar(root, orient="horizontal", mode="determinate", style='blue.Horizontal.TProgressbar')
style.configure('blue.Horizontal.TProgressbar', troughcolor='#e6e6e6', bordercolor='#e6e6e6',
                background=primary_color, lightcolor=primary_color, darkcolor=primary_color)

# Frames for better layout
main_frame = ttk.Frame(root, padding=10, style='TFrame')
main_frame.pack(fill='both', expand=True)

# Divide the main_frame into left and right frames (each 50% width)
left_frame = ttk.Frame(main_frame, padding=10, style='TFrame')
left_frame.pack(side='left', fill='both', expand=True)

right_frame = ttk.Frame(main_frame, padding=10, style='TFrame')
right_frame.pack(side='right', fill='both', expand=True)

# Frames inside left_frame
controls_frame = ttk.Frame(left_frame, padding=10, style='TFrame')
controls_frame.pack(fill='both', expand=True)

progress_frame = ttk.Frame(left_frame, padding=10, style='TFrame')
progress_frame.pack(fill='x')

# Frames inside right_frame
top_frame = ttk.Frame(right_frame, padding=10, style='TFrame')
top_frame.pack(fill='x')

info_frame = ttk.Frame(right_frame, padding=10, style='TFrame')
info_frame.pack(fill='both', expand=True)

media_frame = ttk.Frame(right_frame, padding=10, style='TFrame')
media_frame.pack(fill='x')

# Butterworth Filter functions for low, mid, high bands
def butter_band_filter(data, cutoff, fs, btype, order=5):
    nyquist = 0.5 * fs
    if isinstance(cutoff, list):
        normal_cutoff = [c / nyquist for c in cutoff]
    else:
        normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)

# Compression function
def apply_compression(audio, threshold_db, ratio):
    threshold = 10 ** (threshold_db / 20)
    compressed_audio = np.copy(audio)
    over_threshold = np.abs(audio) > threshold
    compressed_audio[over_threshold] = np.sign(audio[over_threshold]) * (
        threshold + (np.abs(audio[over_threshold]) - threshold) / ratio
    )
    return compressed_audio

# File selection and display functions
def choose_file():
    global audio_path
    try:
        audio_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac")])
        if audio_path:
            logging.info(f"Selected file: {audio_path}")
            selected_file_label.config(text=f"Selected File: {os.path.basename(audio_path)}")
            display_info()
            analyze_audio()
    except Exception as e:
        logging.error("Error choosing file: %s", e)
        messagebox.showerror("Error", "Error choosing file.")

# Display audio file information
def display_info():
    global audio_path, audio_data, sr, duration
    if audio_path:
        try:
            audio_data, sr = sf.read(audio_path, always_2d=True)
            duration = len(audio_data) / sr

            # Update total duration
            mins, secs = divmod(duration, 60)
            total_duration.set(f"{int(mins):02d}:{int(secs):02d}")

            # Clear previous info
            for widget in info_frame.winfo_children():
                widget.destroy()

            # Display audio info
            ttk.Label(info_frame, text=f"Sample Rate: {sr} Hz", font=("Helvetica", 12)).pack(anchor='w', pady=5)
            channels = 'Mono' if audio_data.shape[1] == 1 else f'{audio_data.shape[1]} Channels'
            ttk.Label(info_frame, text=f"Channels: {channels}", font=("Helvetica", 12)).pack(anchor='w', pady=5)
            ttk.Label(info_frame, text=f"Duration: {duration:.2f} seconds", font=("Helvetica", 12)).pack(anchor='w', pady=5)

            # Show waveform
            show_waveform(audio_data, sr)

            # Reset playback position
            global playback_position
            playback_position = 0

            # Update position slider
            position_slider.config(from_=0, to=duration)
            position_slider.set(0)
        except Exception as e:
            logging.error("Error loading audio file: %s", e)
            messagebox.showerror("Error", "Error loading audio file.")

# Function to show waveform
def show_waveform(data, sample_rate):
    # Convert stereo to mono if necessary
    if data.shape[1] > 1:
        data = np.mean(data, axis=1)

    fig = plt.Figure(figsize=(8, 2), dpi=100)
    ax = fig.add_subplot(111)
    times = np.arange(len(data)) / sample_rate
    ax.plot(times, data)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform')
    ax.set_xlim([0, times[-1]])

    # Convert plot to image and display in Tkinter
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    img_tk = ImageTk.PhotoImage(img)
    waveform_label = ttk.Label(info_frame, image=img_tk)
    waveform_label.image = img_tk  # Keep a reference to prevent garbage collection
    waveform_label.pack(pady=5)

# Analyze audio and display recommendations
def analyze_audio():
    global audio_data, sr
    if audio_data is None:
        logging.warning("No audio file loaded to analyze.")
        messagebox.showwarning("Warning", "Please select an audio file first.")
        return

    try:
        audio_mono = np.mean(audio_data, axis=1)

        # Calculate RMS amplitude
        rms_amplitude = np.sqrt(np.mean(audio_mono**2))
        peak_amplitude = np.max(np.abs(audio_mono))

        # Frequency analysis
        freqs, power_spectrum = calculate_power_spectrum(audio_mono, sr)

        # Display more audio info
        ttk.Label(info_frame, text=f"Peak Amplitude: {peak_amplitude:.4f}", font=("Helvetica", 12)).pack(anchor='w', pady=5)
        ttk.Label(info_frame, text=f"RMS Amplitude: {rms_amplitude:.4f}", font=("Helvetica", 12)).pack(anchor='w', pady=5)

        # Show power spectrum
        show_power_spectrum(freqs, power_spectrum)

        # Automatically adjust mastering settings based on analysis
        adjust_mastering_settings(audio_mono, rms_amplitude, peak_amplitude)

        logging.info("Audio analysis complete.")
    except Exception as e:
        logging.error("Error analyzing audio: %s", e)
        messagebox.showerror("Error", "Error analyzing audio.")

def calculate_power_spectrum(data, sample_rate):
    # Perform FFT
    fft_result = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
    power_spectrum = np.abs(fft_result)
    return freqs, power_spectrum

def show_power_spectrum(freqs, power_spectrum):
    fig = plt.Figure(figsize=(8, 2), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(freqs, power_spectrum)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Power Spectrum')
    ax.set_xlim([0, np.max(freqs)])
    ax.set_ylim([0, np.max(power_spectrum)])

    # Convert plot to image and display in Tkinter
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    img_tk = ImageTk.PhotoImage(img)
    spectrum_label = ttk.Label(info_frame, image=img_tk)
    spectrum_label.image = img_tk  # Keep a reference to prevent garbage collection
    spectrum_label.pack(pady=5)

def adjust_mastering_settings(audio_mono, rms_amplitude, peak_amplitude):
    # Analyze frequency bands
    low_band = butter_band_filter(audio_mono, 200, sr, btype='low')
    mid_band = butter_band_filter(audio_mono, [200, 5000], sr, btype='band')
    high_band = butter_band_filter(audio_mono, 5000, sr, btype='high')

    low_rms = np.sqrt(np.mean(low_band**2))
    mid_rms = np.sqrt(np.mean(mid_band**2))
    high_rms = np.sqrt(np.mean(high_band**2))

    # Automatically set EQ gains
    low_gain.set(1.0)
    mid_gain.set(1.0)
    high_gain.set(1.0)

    if low_rms < 0.1 * rms_amplitude:
        low_gain.set(1.2)
    if high_rms > 2 * mid_rms:
        high_gain.set(0.8)

    # Set compression settings
    if rms_amplitude < 0.2:
        compression_threshold.set(-20)
        compression_ratio.set(4.0)
    else:
        compression_threshold.set(-15)
        compression_ratio.set(2.5)

    # Adjust target loudness
    target_loudness.set(-14)

    # Adjust noise reduction strength
    noise_reduction_strength.set(1.0)

# Mastering function
def master_audio():
    global audio_data, sr
    if audio_data is None:
        logging.warning("No audio file loaded to master.")
        messagebox.showwarning("Warning", "Please select an audio file first.")
        return

    def _master():
        try:
            progress['value'] = 0
            progress.update()

            # Copy the audio data to avoid modifying the original
            mastered_audio = np.copy(audio_data)
            num_steps = 4  # Total number of processing steps
            step = 0

            # Apply Noise Reduction
            if apply_noise_reduction.get():
                logging.info("Applying noise reduction...")
                for channel in range(mastered_audio.shape[1]):
                    mastered_audio[:, channel] = nr.reduce_noise(
                        y=mastered_audio[:, channel],
                        sr=sr,
                        prop_decrease=noise_reduction_strength.get()
                    )
                step += 1
                progress['value'] = (step / num_steps) * 100
                progress.update()

            # Apply EQ
            if apply_eq.get():
                logging.info("Applying equalization...")
                for channel in range(mastered_audio.shape[1]):
                    low_adj = butter_band_filter(
                        mastered_audio[:, channel], 200, sr, btype='low'
                    ) * low_gain.get()
                    mid_adj = butter_band_filter(
                        mastered_audio[:, channel], [200, 5000], sr, btype='band'
                    ) * mid_gain.get()
                    high_adj = butter_band_filter(
                        mastered_audio[:, channel], 5000, sr, btype='high'
                    ) * high_gain.get()
                    mastered_audio[:, channel] = low_adj + mid_adj + high_adj
                step += 1
                progress['value'] = (step / num_steps) * 100
                progress.update()

            # Apply Compression
            if apply_compression_var.get():
                logging.info("Applying compression...")
                threshold = compression_threshold.get()
                ratio = compression_ratio.get()
                for channel in range(mastered_audio.shape[1]):
                    mastered_audio[:, channel] = apply_compression(
                        mastered_audio[:, channel], threshold, ratio
                    )
                step += 1
                progress['value'] = (step / num_steps) * 100
                progress.update()

            # Apply Loudness Normalization
            if apply_loudness_norm.get():
                logging.info("Applying loudness normalization...")
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(mastered_audio)
                mastered_audio = pyln.normalize.loudness(
                    mastered_audio, loudness, target_loudness.get()
                )
                step += 1
                progress['value'] = (step / num_steps) * 100
                progress.update()

            # Save the mastered audio
            output_path = filedialog.asksaveasfilename(
                defaultextension=".wav", filetypes=[("WAV files", "*.wav")]
            )
            if output_path:
                sf.write(output_path, mastered_audio, sr)
                logging.info(f"Mastered audio saved to {output_path}")
                progress['value'] = 100
                progress.update()
                update_queue.put(lambda: messagebox.showinfo("Success", f"Mastered audio saved to {output_path}"))
            else:
                logging.info("Mastering canceled by user.")
                progress['value'] = 0
                progress.update()
        except Exception as e:
            logging.error("Error during mastering: %s", e)
            update_queue.put(lambda: messagebox.showerror("Error", "An error occurred during mastering."))
            progress['value'] = 0
            progress.update()

    threading.Thread(target=_master).start()

# Media Player Controls using Threading
def toggle_play_pause():
    if is_playing:
        pause_audio()
    else:
        play_audio()

def play_audio():
    global is_playing, playback_thread
    if audio_data is None:
        logging.warning("No audio file loaded to play.")
        messagebox.showwarning("Warning", "Please select an audio file first.")
        return
    if not is_playing:
        is_playing = True
        play_pause_button.config(image=pause_icon)
        playback_thread = threading.Thread(target=playback_loop, daemon=True)
        playback_thread.start()
        update_playback_position()
        logging.info("Playback started.")

def pause_audio():
    global is_playing
    if is_playing:
        is_playing = False
        play_pause_button.config(image=play_icon)
        logging.info("Playback paused.")

def stop_audio():
    global is_playing, playback_position
    if is_playing:
        is_playing = False
        logging.info("Playback stopped.")
    playback_position = 0
    position_slider.set(0)
    elapsed_time.set("00:00")
    play_pause_button.config(image=play_icon)

def playback_loop():
    global playback_position, is_playing, pyaudio_instance, stream
    try:
        with playback_lock:
            data = audio_data
            fs = sr

            # Prepare audio data
            audio = data * volume.get()
            audio_int16 = (audio * 32767).astype(np.int16)
            num_channels = data.shape[1]

            # Initialize PyAudio
            pyaudio_instance = pyaudio.PyAudio()

            def callback(in_data, frame_count, time_info, status):
                global playback_position
                if not is_playing or playback_position >= len(audio_int16):
                    return (None, pyaudio.paComplete)

                end_position = min(playback_position + frame_count, len(audio_int16))
                frames = audio_int16[playback_position:end_position]

                # Update playback position
                playback_position = end_position

                # Update the position slider and time display
                current_time = playback_position / fs
                root.after(0, update_playback_position)

                # Interleave channels if necessary
                if num_channels > 1:
                    frames = frames.reshape(-1)
                else:
                    frames = frames.flatten()

                return (frames.tobytes(), pyaudio.paContinue)

            stream = pyaudio_instance.open(format=pyaudio.paInt16,
                                           channels=num_channels,
                                           rate=fs,
                                           output=True,
                                           stream_callback=callback)

            stream.start_stream()

            # Keep the thread alive while the stream is active
            while stream.is_active():
                time.sleep(0.1)

            stream.stop_stream()
            stream.close()
            pyaudio_instance.terminate()
            is_playing = False
            root.after(0, play_pause_button.config, {'image': play_icon})

    except Exception as e:
        logging.error("Error during playback: %s", e)
        update_queue.put(lambda: messagebox.showerror("Playback Error", f"An error occurred during playback:\n{e}"))
        is_playing = False
        root.after(0, play_pause_button.config, {'image': play_icon})

def update_playback_position():
    if is_playing:
        current_time = playback_position / sr
        mins, secs = divmod(current_time, 60)
        elapsed_time.set(f"{int(mins):02d}:{int(secs):02d}")
        position_slider.set(current_time)
        root.after(500, update_playback_position)
    else:
        play_pause_button.config(image=play_icon)

def set_playback_position(value):
    global playback_position, is_playing
    if audio_data is None:
        return
    with playback_lock:
        new_position = int(float(value) * sr)
        playback_position = new_position
        current_time = playback_position / sr
        mins, secs = divmod(current_time, 60)
        elapsed_time.set(f"{int(mins):02d}:{int(secs):02d}")
        if is_playing:
            # Restart playback from the new position
            pause_audio()
            play_audio()

def set_volume(value):
    volume.set(float(value))

# Update the UI in a thread-safe manner
def process_queue():
    try:
        while not update_queue.empty():
            func = update_queue.get()
            func()
    except Exception as e:
        logging.error("Error processing queue: %s", e)
    finally:
        root.after(100, process_queue)

process_queue()

# GUI Layout
# Audio File Controls (in top_frame of right_frame)
choose_button = ttk.Button(top_frame, text="Choose Audio File", command=choose_file, style='Normal.TButton')
choose_button.pack(side='left', padx=10, pady=10)
start_button = ttk.Button(top_frame, text="Start Mastering", command=master_audio, style='Normal.TButton')
start_button.pack(side='left', padx=10, pady=10)

# Selected File Label (will be updated when a file is chosen)
selected_file_label = ttk.Label(top_frame, text="No file selected.", font=("Helvetica", 14))
selected_file_label.pack(pady=10)

# Info Frame (now under Selected File Label)
# The info_frame will be populated when an audio file is loaded

# Media Player Controls (Below the Info Frame in right_frame)
media_box = ttk.LabelFrame(media_frame, text="Media Player", padding=10, style='TFrame')
media_box.pack(fill='x', pady=10)

# Variables for media player GUI
elapsed_time.set("00:00")
total_duration.set("00:00")

# Time Labels
time_frame = ttk.Frame(media_box, style='TFrame')
time_frame.pack(fill='x')

elapsed_label = ttk.Label(time_frame, textvariable=elapsed_time)
elapsed_label.pack(side='left')

remaining_label = ttk.Label(time_frame, textvariable=total_duration)
remaining_label.pack(side='right')

# Playback Position Slider
position_slider = ttk.Scale(media_box, from_=0, to=1, orient=tk.HORIZONTAL, length=600, command=lambda v: set_playback_position(v))
position_slider.pack(pady=5, fill='x')

# Control Buttons
control_frame = ttk.Frame(media_box, style='TFrame')
control_frame.pack(pady=5)

if play_icon and pause_icon and stop_icon:
    play_pause_button = tk.Button(control_frame, image=play_icon, command=toggle_play_pause, bd=0)
    play_pause_button.pack(side='left', padx=5)
    stop_button = tk.Button(control_frame, image=stop_icon, command=stop_audio, bd=0)
    stop_button.pack(side='left', padx=5)
else:
    play_pause_button = ttk.Button(control_frame, text="Play/Pause", command=toggle_play_pause)
    play_pause_button.pack(side='left', padx=5)
    stop_button = ttk.Button(control_frame, text="Stop", command=stop_audio)
    stop_button.pack(side='left', padx=5)

# Volume Control
volume_frame = ttk.Frame(media_box, style='TFrame')
volume_frame.pack(pady=5, fill='x')

volume_label = ttk.Label(volume_frame, text="Volume")
volume_label.pack(side='left')

volume_slider = ttk.Scale(volume_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=200, command=lambda v: set_volume(v))
volume_slider.set(volume.get())
volume_slider.pack(side='left', padx=5)

# Progress Bar
progress.pack(fill="x", pady=5, in_=progress_frame)

# Mastering Settings (in controls_frame of left_frame)
controls_box = ttk.LabelFrame(controls_frame, text="Mastering Controls", padding=20, style='TFrame')
controls_box.pack(fill='both', expand=True)

# Align labels and controls using grid
sliders = [
    ("Noise Reduction Strength", noise_reduction_strength, 0.0, 2.0),
    ("Target Loudness (LUFS)", target_loudness, -30, -5),
    ("Low Gain", low_gain, 0.5, 2.0),
    ("Mid Gain", mid_gain, 0.5, 2.0),
    ("High Gain", high_gain, 0.5, 2.0),
    ("Compression Threshold (dB)", compression_threshold, -60, 0),
    ("Compression Ratio", compression_ratio, 1.0, 10.0)
]

for idx, (label_text, var, min_val, max_val) in enumerate(sliders):
    ttk.Label(controls_box, text=label_text, font=("Helvetica", 12)).grid(row=idx, column=0, padx=5, pady=5, sticky='e')
    ttk.Scale(controls_box, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var, length=300).grid(row=idx, column=1, padx=5, pady=5, sticky='w')

# Apply Effects Box (Aligned using grid)
apply_effects_box = ttk.LabelFrame(controls_frame, text="Apply Effects", padding=20, style='TFrame')
apply_effects_box.pack(fill='x', pady=10)

# Add checkboxes with descriptions (Aligned using grid)
effects = [
    ("Apply Noise Reduction", apply_noise_reduction, "Reduces background noise from the audio."),
    ("Apply EQ", apply_eq, "Adjusts the balance of frequency components."),
    ("Apply Compression", apply_compression_var, "Reduces the dynamic range of the audio."),
    ("Apply Loudness Normalization", apply_loudness_norm, "Adjusts the overall loudness to a target level."),
]

for idx, (effect_name, effect_var, effect_desc) in enumerate(effects):
    ttk.Checkbutton(apply_effects_box, text=effect_name, variable=effect_var).grid(row=idx, column=0, sticky='w', padx=5, pady=5)
    ttk.Label(apply_effects_box, text=effect_desc, font=('Helvetica', 10), foreground='#555555').grid(row=idx, column=1, sticky='w', padx=5, pady=5)

# Start the main loop
root.mainloop()
