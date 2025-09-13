# sound_alert.py
import winsound

def play_warning_sound():
    """Plays a short beep for warnings"""
    winsound.Beep(1000, 500)  # 1000 Hz frequency, 500ms duration
