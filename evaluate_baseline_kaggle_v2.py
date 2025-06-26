import torch
from transformers import LLaVAModel, LLaVAProcessor
from load_llava_kaggle_v2 import prepare_data_loader, Config
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

# Load model and data
processor = LLaVAProcessor.from_pretrained(Config.MODEL_NAME)
model = LLaVAModel.from_pretrained(Config.MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data_loader = prepare_data_loader()

def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_inputs, batch_answers in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch_inputs["input_ids"].to(device)
            pixel_values = batch_inputs["pixel_values"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            logits = outputs.logits  # Adjust per LLaVA output
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            # Process answers (simplified to indices for now)
            labels = [processor.tokenizer.encode(ans.lower(), add_special_tokens=False)[0] for ans in batch_answers]
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    try:
        accuracy, precision, recall, f1 = evaluate_model(model, data_loader)
        print(f"Baseline Evaluation: Acc={accuracy * 100:.2f}%, P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    except Exception as e:
        print(f"Evaluation error: {e}")
