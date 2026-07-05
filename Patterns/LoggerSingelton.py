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

                instance = super(LoggerSingleton, cls).__new__(cls)

                os.makedirs(log_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if log_file is None:
                    log_file = f"automation_{timestamp}.log"

                logger = logging.getLogger("AutomationLogger")
                logger.setLevel(logging.DEBUG)

                if not logger.handlers:
                    file_path = os.path.join(log_dir, log_file)

                    file_handler = logging.FileHandler(
                        file_path,
                        mode="w",
                        encoding="utf-8"
                    )
                    file_handler.setLevel(logging.DEBUG)

                    console_handler = logging.StreamHandler()
                    console_handler.setLevel(logging.DEBUG)

                    formatter = logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                    )

                    file_handler.setFormatter(formatter)
                    console_handler.setFormatter(formatter)

                    logger.addHandler(file_handler)
                    logger.addHandler(console_handler)

                    instance.file_handler = file_handler

                logger.propagate = False
                instance.logger = logger

                cls._instance = instance

        return cls._instance

    def get_logger(self):
        return self.logger


logger_instance = LoggerSingleton().get_logger()


def printer(type, value, **kwargs):
    type = type.lower()
    if type == "debug":
        logger_instance.debug(value, **kwargs)
    elif type == "info":
        logger_instance.info(value, **kwargs)
    elif type == "warning":
        logger_instance.warning(value, **kwargs)
    elif type == "error":
        logger_instance.error(value, **kwargs)
    elif type == "critical":
        logger_instance.critical(value, **kwargs)
    else:
        logger_instance.info(f"UNSPECIFIED TYPE: {value}", **kwargs)
