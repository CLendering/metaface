from datasets import load_dataset
import os
import logging
from typing import Optional


def setup_logging(log_file: Optional[str] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        ],
    )


def main(save_dir: str) -> None:
    try:
        logging.info("Loading dataset...")
        ds = load_dataset("HuggingFaceM4/FairFace", "0.25")
        logging.info("Dataset loaded successfully.")

        logging.info(f"Saving dataset to {save_dir}...")
        ds.save_to_disk(save_dir)
        logging.info("Dataset saved successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    save_directory = "datasets/fairface"
    os.makedirs(save_directory, exist_ok=True)
    setup_logging()
    main(save_directory)
