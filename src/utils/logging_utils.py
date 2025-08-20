import logging, os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name="train", level="INFO", log_dir="logs", filename=None):
    log = logging.getLogger(name)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # console
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    # file (rotating)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    filename = filename or "train.log"
    fh = RotatingFileHandler(os.path.join(log_dir, filename), maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log