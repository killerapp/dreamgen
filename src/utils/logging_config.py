import logging
from pathlib import Path


def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """Configure application-wide logging.

    Args:
        log_dir: Directory where log files are stored.
        verbose: If True, log INFO level to console; otherwise only warnings.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[file_handler, console_handler],
    )
