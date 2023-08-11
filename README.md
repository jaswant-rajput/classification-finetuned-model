# text-classification-finetuned-model

Finetuned a distilbert model on emotions dataset from huggingface and uploaded the dataset on huggingfacehub

model_id on huggingface = "jaswant50/distilbert-base-uncased-finetuned-emotion"


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="jaswant50/distilbert-base-uncased-finetuned-emotion")



# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("jaswant50/distilbert-base-uncased-finetuned-emotion")
model = AutoModelForSequenceClassification.from_pretrained("jaswant50/distilbert-base-uncased-finetuned-emotion")
