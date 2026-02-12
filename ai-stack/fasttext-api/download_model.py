import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_PATH = "/models/lid.176.bin"


def download_model():
    """Download fastText language identification model if it doesn't exist"""
    path = Path(MODEL_PATH)
    if path.exists():
        logger.info(f"Model already exists at {MODEL_PATH}")
        file_size = path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"Model size: {file_size:.2f} MB")
        return

    logger.info(f"Downloading model from {MODEL_URL}...")
    logger.info(f"This will be saved to {MODEL_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        file_size = path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"Model downloaded successfully! Size: {file_size:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


if __name__ == "__main__":
    download_model()
