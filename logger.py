# logger.py
import datetime

def log_event(message, level="INFO", logfile="events.log"):
    """Logs events to a file with timestamp and log level."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {level}: {message}"

    # Write to log file
    with open(logfile, "a") as f:
        f.write(log_message + "\n")

    # Also print on console (optional)
    print(log_message)
