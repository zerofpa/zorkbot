import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, ClassLabel

# Load the collected data
with open('collected_data.json', 'r') as f:
    data = json.load(f)

# Convert the list of dictionaries into a dictionary of lists
def convert_to_dict_format(data):
    input_texts = [item['input_text'] for item in data]
    target_texts = [item['target_text'] for item in data]
    return {
        "input_text": input_texts,
        "target_text": target_texts
    }

# Convert data to the required format
data_dict = convert_to_dict_format(data)

# Convert the data into a Hugging Face Dataset
dataset = Dataset.from_dict(data_dict)

# Define class labels based on target texts
unique_labels = list(set(data_dict["target_text"]))
class_labels = ClassLabel(names=unique_labels)

# Map the target texts to integer labels
def map_labels(example):
    example["label"] = class_labels.str2int(example["target_text"])
    return example

dataset = dataset.map(map_labels, remove_columns=["target_text"])

# Load the tokenizer for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocess the dataset
def preprocess_function(examples):
    inputs = examples["input_text"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = examples["label"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Convert inputs to float and ensure labels are long
def convert_types(examples):
    examples["input_ids"] = torch.tensor(examples["input_ids"], dtype=torch.float)
    examples["attention_mask"] = torch.tensor(examples["attention_mask"], dtype=torch.float)
    examples["labels"] = torch.tensor(examples["labels"], dtype=torch.long)
    return examples

tokenized_datasets = tokenized_datasets.map(convert_types, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(unique_labels))

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

