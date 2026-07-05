import logging
import os
from datetime import datetime
from threading import Lock


class LoggerSingleton:

    _instance = None
    _lock = Lock()

    def __new__(cls, log_dir="logs", log_file=None):

        with cls._lock:

            if cls._instance is None:

                instance = super().__new__(cls)

                os.makedirs(log_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if not log_file:
                    log_file = f"automation_{timestamp}.log"

                file_path = os.path.join(log_dir, log_file)

                logger = logging.getLogger("AutomationLogger")
                logger.setLevel(logging.DEBUG)

                # 🔥 חשוב: תמיד לנקות handlers ישנים
                logger.handlers.clear()

                formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )

                # file handler
                file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)

                # console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(formatter)

                logger.addHandler(file_handler)
                logger.addHandler(console_handler)

                logger.propagate = False

                instance.logger = logger
                instance.file_path = file_path

                cls._instance = instance

            return cls._instance

    def get_logger(self):
        return self.logger


# =========================
# SIMPLE PRINTER WRAPPER
# =========================
logger_instance = LoggerSingleton().get_logger()


def printer(level, message, **kwargs):

    level = level.lower()

    if level == "debug":
        logger_instance.debug(message, **kwargs)

    elif level == "info":
        logger_instance.info(message, **kwargs)

    elif level == "warning":
        logger_instance.warning(message, **kwargs)

    elif level == "error":
        logger_instance.error(message, **kwargs)

    elif level == "critical":
        logger_instance.critical(message, **kwargs)

    else:
        logger_instance.info(message, **kwargs)