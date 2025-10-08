# logger.py
import datetime

LOG_FILE = "events.log"

def log_event(message, level="INFO"):
    """Logs events to a file with timestamp and log level."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {level}: {message}"

    # Write to log file
    with open(LOG_FILE, "a") as f:
        f.write(log_message + "\n")

    # Also print on console
    print(log_message)