Audio Mastering Tool

An audio mastering application built with Python, providing a graphical user interface (GUI) for processing audio files. The application allows users to apply noise reduction, equalization, compression, and loudness normalization to their audio files. It also features an integrated media player for previewing audio.
Features

    Load and Analyze Audio Files
        Supports WAV, MP3, and FLAC formats.
        Displays audio information: sample rate, channels, duration, peak amplitude, and RMS amplitude.
        Visualizes the waveform and power spectrum of the audio file.

    Mastering Controls
        Noise Reduction: Reduce background noise with adjustable strength.
        Equalization (EQ): Adjust low, mid, and high-frequency gains.
        Compression: Control compression threshold and ratio.
        Loudness Normalization: Set target loudness level (in LUFS).

    Apply Effects
        Selectively apply noise reduction, EQ, compression, and loudness normalization.

    Integrated Media Player
        Play, pause, and stop audio playback.
        Adjust volume and seek within the audio.
        Displays elapsed and total duration.

Installation
Prerequisites

    Python 3.6 or higher

Clone the Repository

bash

git clone https://github.com/yourusername/audio-mastering-tool.git
cd audio-mastering-tool

Install Required Python Packages

Install the required packages using pip:

bash

pip install numpy noisereduce pyloudnorm scipy pillow soundfile pyaudio matplotlib

Note: Installing PyAudio can be tricky on some systems. If you encounter issues, follow the instructions below.
Installing PyAudio on Windows

    Download the appropriate PyAudio wheel file from PyAudio Windows Wheels (choose the one matching your Python version and architecture, e.g., PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl for Python 3.9 x64).

    Install the downloaded wheel file using pip:

    bash

    pip install path_to_downloaded_wheel.whl

Installing PyAudio on Linux

Install the portaudio library and development headers:

bash

sudo apt-get install portaudio19-dev python3-pyaudio

Then install PyAudio:

bash

pip install pyaudio

Installing PyAudio on macOS

Use Homebrew to install portaudio:

bash

brew install portaudio

Then install PyAudio:

bash

pip install pyaudio

Running the Application

Run the Python script:

bash

python audio_mastering_tool.py

Replace audio_mastering_tool.py with the actual name of the Python script if it's different.
Usage

    Load an Audio File
        Click on the "Choose Audio File" button.
        Select an audio file (WAV, MP3, or FLAC).

    View Audio Information
        The selected file's name and audio information will be displayed:
            Sample Rate
            Channels
            Duration
            Peak Amplitude
            RMS Amplitude
        Waveform and power spectrum plots will be shown.

    Adjust Mastering Settings
        Use the sliders under "Mastering Controls" to adjust settings:
            Noise Reduction Strength
            Target Loudness (LUFS)
            Low, Mid, High Gains
            Compression Threshold and Ratio
        Select or deselect effects under "Apply Effects".

    Use the Media Player
        Located on the right side under the audio information.
        Controls:
            Play/Pause: Start or pause playback.
            Stop: Stop playback and reset position.
            Volume Slider: Adjust playback volume.
            Position Slider: Seek within the audio.

    Start Mastering
        Click the "Start Mastering" button.
        A progress bar will indicate processing status.
        After processing, choose a location to save the mastered audio file.

Dependencies

The application depends on the following Python packages:

    numpy
    noisereduce
    pyloudnorm
    scipy
    Pillow
    soundfile
    pyaudio
    matplotlib

Ensure all dependencies are installed using the installation instructions provided.
Notes

    PyAudio Installation: If you have trouble installing PyAudio, refer to the installation instructions in the Installation section.
    Media Player Icons: The application uses play.png, pause.png, and stop.png icons located in a ui directory relative to the script. Ensure these images are available. If not, the application will use text buttons instead.
    Cross-Platform Compatibility: The application should work on Windows, macOS, and Linux systems, provided all dependencies are correctly installed.

Contributing

Contributions are welcome! If you'd like to contribute to this project:

    Fork the repository.
    Create a new branch: git checkout -b feature/YourFeature.
    Commit your changes: git commit -am 'Add your feature'.
    Push to the branch: git push origin feature/YourFeature.
    Submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

    PyAudio for audio playback.
    Noisereduce for noise reduction algorithms.
    PyLoudNorm for loudness normalization.
    Matplotlib for plotting waveforms and spectra.
