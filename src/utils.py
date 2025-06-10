import logging
import time


class Timer:
    def __init__(self, function_name, logger):
        self.function_name = function_name
        self.begin_time = time.time()
        self.logger = logger

    def __del__(self):
        self.logger.info(
            f"{self.function_name} costs {time.time() - self.begin_time:.3f} seconds"
        )


def get_customized_logger():
    logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(filename)10s:%(lineno)4d][%(levelname)5s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger
