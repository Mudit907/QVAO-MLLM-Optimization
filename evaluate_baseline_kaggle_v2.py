import evaluate
from tqdm import tqdm

# Load VQA metric
vqa_metric = evaluate.load("accuracy")

# Evaluate on first 1000 samples
val_dataset = dataset.select(range(1000))

# Evaluate
model.eval()
predictions, references = [], []
for example in tqdm(val_dataset, desc="Evaluating"):
    question = example["question"]
    answer = example["multiple_choice_answer"]
    image = example["image"].convert("RGB")
    
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, max_new_tokens=100)
        pred = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    
    predictions.append(pred)
    references.append(answer)

# Compute accuracy
results = vqa_metric.compute(predictions=predictions, references=references)
print(f"Baseline VQA Accuracy: {results['accuracy']:.4f}")
