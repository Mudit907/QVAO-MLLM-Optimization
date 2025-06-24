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
