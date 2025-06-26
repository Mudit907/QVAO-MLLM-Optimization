import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import LLaVAProcessor, LLaVAModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp

# Configuration
class Config:
    DATA_DIR = "/kaggle/input/vqav2"
    MODEL_NAME = "llava-1.5-7b-hf"
    BATCH_SIZE = 32
    MAX_SAMPLES = 265000  # Full dataset target
    NUM_WORKERS = mp.cpu_count()  # Parallel processing

# Custom Dataset Class
class VQADataset(Dataset):
    def __init__(self, data_dir, max_samples):
        self.dataset = load_dataset("vqav2", split=f"train[:{max_samples}]").to_pandas()
        self.data_dir = data_dir
        self.processor = LLaVAProcessor.from_pretrained(Config.MODEL_NAME)

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            inputs = self.processor(text=row["question"], images=img, return_tensors="pt", padding=True)
            return {k: v.squeeze(0) for k, v in inputs.items()}, row["answer"]
        raise FileNotFoundError(f"Image {img_path} missing")

# Data Loader with Parallel Processing
def prepare_data_loader():
    dataset = VQADataset(Config.DATA_DIR, Config.MAX_SAMPLES)
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, shuffle=True, pin_memory=True)

def save_preprocessed_data(data_loader, save_path="preprocessed_vqav2.pt"):
    inputs, answers = [], []
    for batch_inputs, batch_answers in data_loader:
        inputs.append(batch_inputs)
        answers.append(batch_answers)
    torch.save({"inputs": inputs, "answers": answers}, save_path)
    print(f"Preprocessed {len(answers) * Config.BATCH_SIZE} samples saved to {save_path}")

if __name__ == "__main__":
    try:
        data_loader = prepare_data_loader()
        save_preprocessed_data(data_loader)
    except Exception as e:
        print(f"Data loading failed: {e}")
