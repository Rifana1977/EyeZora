# sound_alert.py
import platform
import os

def play_alert():
    """Plays a buzzer sound depending on the OS."""
    system = platform.system()

    if system == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # frequency=1000Hz, duration=500ms

    elif system == "Darwin":  # macOS
        os.system('afplay /System/Library/Sounds/Glass.aiff')

    else:  # Linux
        os.system('play -nq -t alsa synth 0.5 sine 1000')