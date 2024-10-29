Audio Mastering Tool

A Python-based application with a graphical user interface (GUI) for mastering audio files. 
Apply noise reduction, equalization, compression, and loudness normalization effortlessly. 
Preview your edits using the integrated media player.

![image](https://github.com/user-attachments/assets/23aa3385-b48d-4bf7-ae75-a36f56f055f1)

Installation

Prerequisites

Python: Version 3.6 or higher
FFmpeg: Required for audio processing
Install FFmpeg:
Windows: Download from FFmpeg Downloads and add to your system PATH.
macOS: Use Homebrew:
bash
Code kopieren
brew install ffmpeg
Linux: Use your distribution's package manager:
bash
Code kopieren
sudo apt-get install ffmpeg
Clone the Repository
bash
Code kopieren
git clone https://github.com/Various5/audio-mastering-tool.git
cd audio-mastering-tool
Install Python Dependencies
bash
Code kopieren
pip install numpy noisereduce pyloudnorm scipy Pillow soundfile pyaudio matplotlib pygame
Note: Installing PyAudio may require additional steps:

Windows: Download the appropriate wheel from PyAudio Windows Wheels and install:
bash
Code kopieren
pip install path_to_downloaded_wheel.whl
macOS/Linux: Ensure portaudio is installed:
macOS:
bash
Code kopieren
brew install portaudio
pip install pyaudio
Linux:
bash
Code kopieren
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
Running the Application
bash
Code kopieren
python audio_mastering_tool.py
Replace audio_mastering_tool.py with the actual script name if different.

License
This project is licensed under the MIT License.

Acknowledgments
PyAudio: For audio playback.
Noisereduce: For noise reduction algorithms.
PyLoudNorm: For loudness normalization.
Matplotlib: For plotting waveforms and spectra.
Pillow: For image handling in the GUI.
pygame: For media player functionality.
