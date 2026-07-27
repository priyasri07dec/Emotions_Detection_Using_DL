from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

print("✅ Model Loaded Successfully")