from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
from huggingface_hub import login

# Login to Hugging Face
login(token="your hf token")  # Replace if different

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model in 4-bit quantization
model_id = "llava-hf/llava-1.5-7b-hf"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)

# Test with a sample image (upload "apple.jpg" to Kaggle input)
image = Image.open("/kaggle/input/apple.jpg").convert("RGB")  # Adjust path

# Prompt
prompt = "USER: <image>\nWhat is in the image?\nASSISTANT:"

# Preprocess and generate
inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    generate_ids = model.generate(**inputs, max_new_tokens=100)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print("Output:", output)
