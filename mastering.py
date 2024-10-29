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
import pygame
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ==== Logging Setup ====
logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG to see all logs
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("audio_mastering.log"),
                        logging.StreamHandler(sys.stdout)  # Also log to console
                    ])

# Initialize Pygame Mixer
pygame.mixer.init()

# ==== GUI Setup ====
root = tk.Tk()
root.title("Audio Mastering Tool")
root.geometry("1200x800")
root.minsize(1200, 800)  # Set minimum window size
root.configure(bg='#2b2b2b')  # Dark background for modern look

# Apply a custom style to ttk widgets
style = ttk.Style(root)
style.theme_use('clam')

# Define custom colors for a modern look
primary_color = '#3c3f41'  # Dark gray
secondary_color = '#d3d3d3'  # Light gray
accent_color = '#00bfff'  # Sky blue
background_color = '#2b2b2b'  # Dark background
foreground_color = '#ffffff'  # White text

# Configure styles
style.configure('TFrame', background=background_color)
style.configure('TLabel', background=background_color, foreground=foreground_color, font=('Helvetica', 12))
style.configure('TButton', background=primary_color, foreground=foreground_color, font=('Helvetica', 12), padding=10)
style.configure('TCheckbutton', background=background_color, foreground=foreground_color, font=('Helvetica', 10))
style.configure('Horizontal.TScale', background=background_color)
style.configure('TLabelFrame', background=background_color, foreground=foreground_color)
style.configure('TSeparator', background=foreground_color)
style.configure('Horizontal.TProgressbar', troughcolor=primary_color, background=accent_color, bordercolor=primary_color)

# Create custom styles for buttons
style.map('TButton',
          background=[('active', accent_color), ('!active', primary_color)],
          foreground=[('active', foreground_color)],
          relief=[('pressed', 'sunken'), ('!pressed', 'raised')])

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
is_paused = False
volume = tk.DoubleVar(value=0.5)
elapsed_time = tk.StringVar(value="00:00")
total_duration = tk.StringVar(value="00:00")
duration = 0  # Total duration in seconds
playback_start_time = 0  # Time when playback started
paused_time = 0  # Time when playback was paused

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
progress = ttk.Progressbar(root, orient="horizontal", mode="determinate")
style.configure('blue.Horizontal.TProgressbar', troughcolor=primary_color, bordercolor=primary_color,
                background=accent_color, lightcolor=accent_color, darkcolor=accent_color)

# Frames for better layout
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill='both', expand=True)

# Divide the main_frame into left and right frames (each 50% width)
left_frame = ttk.Frame(main_frame, padding=10)
left_frame.pack(side='left', fill='both', expand=True)

right_frame = ttk.Frame(main_frame, padding=10)
right_frame.pack(side='right', fill='both', expand=True)

# Frames inside left_frame
controls_frame = ttk.Frame(left_frame, padding=10)
controls_frame.pack(fill='both', expand=True)

progress_frame = ttk.Frame(left_frame, padding=10)
progress_frame.pack(fill='x')

# Frames inside right_frame
top_frame = ttk.Frame(right_frame, padding=10)
top_frame.pack(fill='x')

info_frame = ttk.Frame(right_frame, padding=10)
info_frame.pack(fill='both', expand=True)

media_frame = ttk.Frame(right_frame, padding=10)
media_frame.pack(fill='x')

# Initialize waveform_label
waveform_label = None

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
    global audio_path, audio_data, sr, duration, waveform_label, playback_start_time, paused_time
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

            # Show initial waveform
            update_waveform_display(audio_data, sr)

            # Reset playback variables
            stop_audio()
            playback_start_time = 0
            paused_time = 0
            elapsed_time.set("00:00")
            position_slider.set(0)
        except Exception as e:
            logging.error("Error loading audio file: %s", e)
            messagebox.showerror("Error", "Error loading audio file.")

# Function to update waveform display
def update_waveform_display(data, sample_rate):
    global waveform_label

    # Remove the existing waveform label if it exists
    if waveform_label is not None:
        waveform_label.destroy()

    # Convert stereo to mono if necessary
    if data.shape[1] > 1:
        data = np.mean(data, axis=1)

    fig = plt.Figure(figsize=(8, 2), dpi=100)
    ax = fig.add_subplot(111)
    times = np.arange(len(data)) / sample_rate
    ax.plot(times, data, color=accent_color)
    ax.set_xlabel('Time (s)', color=foreground_color)
    ax.set_ylabel('Amplitude', color=foreground_color)
    ax.set_title('Waveform', color=foreground_color)
    ax.set_xlim([0, times[-1]])
    ax.tick_params(axis='x', colors=foreground_color)
    ax.tick_params(axis='y', colors=foreground_color)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(primary_color)

    # Convert plot to image and display in Tkinter
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    img_tk = ImageTk.PhotoImage(img)
    waveform_label = ttk.Label(info_frame, image=img_tk)
    waveform_label.image = img_tk  # Keep a reference to prevent garbage collection
    waveform_label.pack(pady=5, fill='both', expand=True)

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
    ax.plot(freqs, power_spectrum, color=accent_color)
    ax.set_xlabel('Frequency (Hz)', color=foreground_color)
    ax.set_ylabel('Magnitude', color=foreground_color)
    ax.set_title('Power Spectrum', color=foreground_color)
    ax.set_xlim([0, np.max(freqs)])
    ax.set_ylim([0, np.max(power_spectrum)])
    ax.tick_params(axis='x', colors=foreground_color)
    ax.tick_params(axis='y', colors=foreground_color)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(primary_color)

    # Convert plot to image and display in Tkinter
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    img_tk = ImageTk.PhotoImage(img)
    spectrum_label = ttk.Label(info_frame, image=img_tk)
    spectrum_label.image = img_tk  # Keep a reference to prevent garbage collection
    spectrum_label.pack(pady=5, fill='both', expand=True)

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

# Preview Changes function
def preview_changes():
    global audio_data, sr
    if audio_data is None:
        logging.warning("No audio file loaded to preview.")
        messagebox.showwarning("Warning", "Please select an audio file first.")
        return

    def _preview():
        try:
            # Copy the audio data to avoid modifying the original
            preview_audio = np.copy(audio_data)

            # Apply Noise Reduction
            if apply_noise_reduction.get():
                logging.info("Applying noise reduction for preview...")
                for channel in range(preview_audio.shape[1]):
                    preview_audio[:, channel] = nr.reduce_noise(
                        y=preview_audio[:, channel],
                        sr=sr,
                        prop_decrease=noise_reduction_strength.get()
                    )

            # Apply EQ
            if apply_eq.get():
                logging.info("Applying equalization for preview...")
                for channel in range(preview_audio.shape[1]):
                    low_adj = butter_band_filter(
                        preview_audio[:, channel], 200, sr, btype='low'
                    ) * low_gain.get()
                    mid_adj = butter_band_filter(
                        preview_audio[:, channel], [200, 5000], sr, btype='band'
                    ) * mid_gain.get()
                    high_adj = butter_band_filter(
                        preview_audio[:, channel], 5000, sr, btype='high'
                    ) * high_gain.get()
                    preview_audio[:, channel] = low_adj + mid_adj + high_adj

            # Apply Compression
            if apply_compression_var.get():
                logging.info("Applying compression for preview...")
                threshold = compression_threshold.get()
                ratio = compression_ratio.get()
                for channel in range(preview_audio.shape[1]):
                    preview_audio[:, channel] = apply_compression(
                        preview_audio[:, channel], threshold, ratio
                    )

            # Apply Loudness Normalization
            if apply_loudness_norm.get():
                logging.info("Applying loudness normalization for preview...")
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(preview_audio)
                preview_audio = pyln.normalize.loudness(
                    preview_audio, loudness, target_loudness.get()
                )

            # Update the waveform display
            update_waveform_display(preview_audio, sr)
            logging.info("Preview updated.")
        except Exception as e:
            logging.error("Error during preview: %s", e)
            messagebox.showerror("Error", "An error occurred during preview.")

    threading.Thread(target=_preview).start()

# Optional: Bind mastering controls to update waveform automatically
def on_mastering_control_change(*args):
    # Debounce to prevent excessive updates
    if hasattr(on_mastering_control_change, 'after_id'):
        root.after_cancel(on_mastering_control_change.after_id)
    on_mastering_control_change.after_id = root.after(500, preview_changes)

# Bind variables
for var in [
    noise_reduction_strength,
    low_gain,
    mid_gain,
    high_gain,
    compression_threshold,
    compression_ratio,
    target_loudness,
    apply_noise_reduction,
    apply_eq,
    apply_compression_var,
    apply_loudness_norm
]:
    var.trace_add('write', on_mastering_control_change)

# Media Player Controls using pygame
def toggle_play_pause():
    global is_playing, is_paused, playback_thread, playback_stop_event, playback_pause_event
    if is_playing:
        if not is_paused:
            pause_audio()
        else:
            resume_audio()
    else:
        play_audio()

def play_audio():
    global is_playing, is_paused, playback_thread, playback_stop_event, playback_pause_event, playback_start_time
    if audio_path is None:
        logging.warning("No audio file loaded to play.")
        messagebox.showwarning("Warning", "Please select an audio file first.")
        return

    is_playing = True
    is_paused = False
    play_pause_button.config(image=pause_icon)
    playback_stop_event.clear()
    playback_pause_event.clear()
    playback_thread = threading.Thread(target=playback_loop, daemon=True)
    playback_thread.start()
    playback_start_time = time.time() - playback_position.get()
    logging.info("Playback started.")

def playback_loop():
    global is_playing, is_paused, playback_start_time
    try:
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.set_volume(volume.get())
        pygame.mixer.music.play(start=playback_position.get())

        while is_playing:
            if playback_stop_event.is_set():
                pygame.mixer.music.stop()
                break
            if playback_pause_event.is_set():
                pygame.mixer.music.pause()
                is_paused = True
                play_pause_button.config(image=play_icon)
                while playback_pause_event.is_set():
                    time.sleep(0.1)
                pygame.mixer.music.unpause()
                is_paused = False
                play_pause_button.config(image=pause_icon)
                playback_start_time = time.time() - playback_position.get()

            # Update playback position
            current_time = pygame.mixer.music.get_pos() / 1000.0  # get_pos returns milliseconds
            playback_position.set(current_time)
            mins, secs = divmod(current_time, 60)
            elapsed_time.set(f"{int(mins):02d}:{int(secs):02d}")
            time.sleep(0.5)

            # Check if playback has finished
            if not pygame.mixer.music.get_busy():
                break

        # Playback finished
        is_playing = False
        is_paused = False
        play_pause_button.config(image=play_icon)
        playback_position.set(0)
        elapsed_time.set("00:00")
        logging.info("Playback finished.")
    except Exception as e:
        logging.error("Error during playback: %s", e)
        messagebox.showerror("Error", "An error occurred during playback.")
        is_playing = False
        is_paused = False
        play_pause_button.config(image=play_icon)

def pause_audio():
    global is_paused
    if is_playing and not is_paused:
        is_paused = True
        playback_pause_event.set()
        play_pause_button.config(image=play_icon)
        logging.info("Playback paused.")

def resume_audio():
    global is_paused
    if is_playing and is_paused:
        is_paused = False
        playback_pause_event.clear()
        play_pause_button.config(image=pause_icon)
        logging.info("Playback resumed.")

def stop_audio():
    global is_playing, is_paused
    if is_playing:
        is_playing = False
        is_paused = False
        playback_stop_event.set()
        play_pause_button.config(image=play_icon)
        playback_position.set(0)
        elapsed_time.set("00:00")
        logging.info("Playback stopped.")

def set_playback_position(value):
    global is_playing, is_paused, playback_start_time
    if audio_path is None:
        return
    try:
        new_position = float(value)
        playback_position.set(new_position)
        if is_playing:
            pygame.mixer.music.stop()
            pygame.mixer.music.play(start=new_position)
            playback_start_time = time.time() - new_position
            logging.info(f"Playback position set to {new_position:.3f} seconds.")
    except Exception as e:
        logging.error("Error setting playback position: %s", e)

def set_volume_control(value):
    volume.set(float(value))
    pygame.mixer.music.set_volume(volume.get())

# Playback control variables
playback_position = tk.DoubleVar(value=0.0)
playback_stop_event = threading.Event()
playback_pause_event = threading.Event()

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
choose_button = ttk.Button(top_frame, text="Choose Audio File", command=choose_file)
choose_button.pack(side='left', padx=10, pady=10)
start_button = ttk.Button(top_frame, text="Start Mastering", command=master_audio)
start_button.pack(side='left', padx=10, pady=10)

# Selected File Label (will be updated when a file is chosen)
selected_file_label = ttk.Label(top_frame, text="No file selected.", font=("Helvetica", 14))
selected_file_label.pack(pady=10, fill='x')

# Media Player Controls (Below the Info Frame in right_frame)
media_box = ttk.LabelFrame(media_frame, text="Media Player", padding=10)
media_box.pack(fill='x', pady=10)

# Time Labels
time_frame = ttk.Frame(media_box)
time_frame.pack(fill='x')

elapsed_label = ttk.Label(time_frame, textvariable=elapsed_time)
elapsed_label.pack(side='left')

remaining_label = ttk.Label(time_frame, textvariable=total_duration)
remaining_label.pack(side='right')

# Playback Position Slider
position_slider = ttk.Scale(media_box, from_=0, to=1, orient=tk.HORIZONTAL, length=600, variable=playback_position, command=lambda v: set_playback_position(v))
position_slider.pack(pady=5, fill='x')

# Control Buttons
control_frame = ttk.Frame(media_box)
control_frame.pack(pady=5)

if play_icon and pause_icon and stop_icon:
    play_pause_button = tk.Button(control_frame, image=play_icon, command=toggle_play_pause, bd=0, bg=background_color, activebackground=primary_color)
    play_pause_button.pack(side='left', padx=5)
    stop_button = tk.Button(control_frame, image=stop_icon, command=stop_audio, bd=0, bg=background_color, activebackground=primary_color)
    stop_button.pack(side='left', padx=5)
else:
    play_pause_button = ttk.Button(control_frame, text="Play/Pause", command=toggle_play_pause)
    play_pause_button.pack(side='left', padx=5)
    stop_button = ttk.Button(control_frame, text="Stop", command=stop_audio)
    stop_button.pack(side='left', padx=5)

# Volume Control
volume_frame = ttk.Frame(media_box)
volume_frame.pack(pady=5, fill='x')

volume_label = ttk.Label(volume_frame, text="Volume")
volume_label.pack(side='left')
volume_slider = ttk.Scale(volume_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=200, variable=volume, command=lambda v: set_volume_control(v))
volume_slider.set(volume.get())
volume_slider.pack(side='left', padx=5)

# Progress Bar
progress.pack(fill="x", pady=5, in_=progress_frame)

# Mastering Settings (in controls_frame of left_frame)
controls_box = ttk.LabelFrame(controls_frame, text="Mastering Controls", padding=20)
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
    scale = ttk.Scale(controls_box, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var, length=300)
    scale.grid(row=idx, column=1, padx=5, pady=5, sticky='we')
    controls_box.columnconfigure(1, weight=1)
    # Display the current value
    value_label = ttk.Label(controls_box, textvariable=var, width=5)
    value_label.grid(row=idx, column=2, padx=5, pady=5, sticky='w')

# Apply Effects Box (Aligned using grid)
apply_effects_box = ttk.LabelFrame(controls_frame, text="Apply Effects", padding=20)
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
    ttk.Label(apply_effects_box, text=effect_desc, font=('Helvetica', 10), foreground=secondary_color).grid(row=idx, column=1, sticky='w', padx=5, pady=5)
    apply_effects_box.columnconfigure(1, weight=1)

# Preview Changes Button
preview_button = ttk.Button(controls_frame, text="Preview Changes", command=preview_changes)
preview_button.pack(pady=10)

# Start the main loop
root.mainloop()
