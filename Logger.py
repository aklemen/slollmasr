from datetime import datetime

class Logger:
    @staticmethod
    def info(message):
        Logger.log_with_time(f"INFO: {message}")

    @staticmethod
    def warn(message):
        Logger.log_with_time(f"WARNING: {message}")

    @staticmethod
    def log_with_time(message):
        print(f"[{datetime.now()}] {message}")
