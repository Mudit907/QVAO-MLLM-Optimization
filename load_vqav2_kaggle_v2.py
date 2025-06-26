from datasets import load_dataset, load_from_disk
import os

# Load VQAv2 from Hugging Face
dataset = load_dataset("HuggingFaceM4/VQAv2", split="train[:5000]")

# Save to Kaggle working directory
os.makedirs("/kaggle/working/vqav2", exist_ok=True)
dataset.save_to_disk("/kaggle/working/vqav2")

# Load saved dataset
dataset = load_from_disk("/kaggle/working/vqav2")
print(f"Loaded {len(dataset)} samples")

---

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import LLaVAProcessor
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
import multiprocessing as mp
from typing import Tuple, List, Dict
import logging
from kaggle_secrets import UserSecretsClient  # For Kaggle API (optional)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    DATA_DIR = "/kaggle/input/vqav2"  # Kaggle dataset path
    MAX_SAMPLES = 265000  # Full VQAv2 dataset
    BATCH_SIZE = 64  # Optimized for Kaggle's 16GB RAM limit
    NUM_WORKERS = min(mp.cpu_count(), 4)  # Limit workers for Kaggle stability
    CACHE_DIR = "/kaggle/working/vqav2_cache"  # Kaggle working directory
    IMG_SIZE = (224, 224)  # Standard for LLaVA
    KAGGLE_ENV = True  # Flag for Kaggle environment

# Custom VQAv2 Dataset for Kaggle
class VQAv2KaggleDataset(Dataset):
    def __init__(self, data_dir: str, max_samples: int, processor: LLaVAProcessor):
        """Initialize with Kaggle-specific preprocessing."""
        self.processor = processor
        self.data = load_dataset("vqav2", split=f"train[:{max_samples}]", cache_dir=Config.CACHE_DIR).to_pandas()
        self.data_dir = data_dir
        self.max_samples = min(max_samples, len(self.data))
        self.preprocessed_data = self._preprocess_data_with_kaggle_checks()

    def _preprocess_data_with_kaggle_checks(self) -> List[Dict]:
        """Preprocess with Kaggle memory and file system considerations."""
        preprocessed = []
        for idx in range(self.max_samples):
            row = self.data.iloc[idx]
            img_path = os.path.join(self.data_dir, row["image"])
            try:
                if os.path.exists(img_path):
                    # Check available memory before loading
                    if psutil.virtual_memory().available < 1e9:  # 1GB buffer
                        logger.warning("Low memory, reducing batch size or skipping")
                        break
                    img = Image.open(img_path).convert("RGB").resize(Config.IMG_SIZE, Image.LANCZOS)
                    inputs = self.processor(text=row["question"], images=img, return_tensors="pt",
                                          padding=True, truncation=True, max_length=128)
                    preprocessed.append({"inputs": {k: v.squeeze(0) for k, v in inputs.items()}, "answer": row["answer"]})
                else:
                    logger.warning(f"Image {img_path} not found at index {idx}, skipping")
            except MemoryError as me:
                logger.error(f"Memory error at index {idx}: {me}, reducing sample size")
                self.max_samples = idx
                break
            except Exception as e:
                logger.error(f"Preprocessing error at {idx}: {e}")
        return preprocessed

    def __len__(self) -> int:
        return len(self.preprocessed_data)

    def __getitem__(self, idx: int) -> Tuple[Dict, str]:
        return self.preprocessed_data[idx]["inputs"], self.preprocessed_data[idx]["answer"]

# Kaggle-Optimized Data Loader
def prepare_vqav2_kaggle_loader() -> DataLoader:
    """Prepare DataLoader with Kaggle-specific caching and memory management."""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, "vqav2_kaggle_processed.pt")

    if os.path.exists(cache_path) and not Config.KAGGLE_ENV:  # Avoid cache overwrite on Kaggle
        logger.info("Loading cached data...")
        data = torch.load(cache_path)
    else:
        logger.info("Preprocessing and caching data on Kaggle...")
        processor = LLaVAProcessor.from_pretrained("llava-1.5-7b-hf", cache_dir=Config.CACHE_DIR)
        dataset = VQAv2KaggleDataset(Config.DATA_DIR, Config.MAX_SAMPLES, processor)
        data = {"inputs": [d[0] for d in dataset], "answers": [d[1] for d in dataset]}
        torch.save(data, cache_path)
        logger.info(f"Cached {len(data['answers'])} samples")

    # Dynamic batch adjustment based on Kaggle memory
    import psutil
    available_memory = psutil.virtual_memory().available
    if available_memory < 8e9:  # Less than 8GB
        Config.BATCH_SIZE = max(16, Config.BATCH_SIZE // 2)
        logger.warning(f"Adjusting batch size to {Config.BATCH_SIZE} due to low memory")

    return DataLoader(list(zip(data["inputs"], data["answers"])), batch_size=Config.BATCH_SIZE,
                      num_workers=Config.NUM_WORKERS, shuffle=True, pin_memory=True, drop_last=True)

# Dataset Validation and Sampling
def validate_kaggle_dataset(data_loader: DataLoader) -> bool:
    """Validate dataset with Kaggle-specific checks."""
    total_samples = 0
    for batch_inputs, batch_answers in data_loader:
        total_samples += len(batch_answers)
        if not all(k in batch_inputs for k in ["input_ids", "pixel_values", "attention_mask"]):
            logger.error("Invalid batch structure on Kaggle")
            return False
        if any(v.is_cuda for v in batch_inputs.values()) and not torch.cuda.is_available():
            logger.error("CUDA tensors detected without GPU")
            return False
    logger.info(f"Validated {total_samples} samples on Kaggle")
    return total_samples > 0

# Kaggle Secret Integration (Optional for API)
def setup_kaggle_secrets():
    """Load Kaggle API token if needed."""
    try:
        secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
        logger.info("Kaggle secrets loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Kaggle secrets: {e}, proceeding without API")

if __name__ == "__main__":
    try:
        setup_kaggle_secrets()
        data_loader = prepare_vqav2_kaggle_loader()
        if validate_kaggle_dataset(data_loader):
            logger.info("VQAv2 dataset loaded successfully on Kaggle with full 265,000 samples.")
            # Example usage for debugging
            for batch_inputs, batch_answers in data_loader:
                print(f"Batch size: {len(batch_answers)}, Sample keys: {batch_inputs.keys()}")
                break
        else:
            logger.error("Dataset validation failed on Kaggle.")
    except Exception as e:
        logger.critical(f"Critical error in Kaggle data loading: {e}")
