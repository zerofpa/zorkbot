import json
import os
import torch
import types
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, ClassLabel

# Load the dataset
with open('/home/xyro/Desktop/adventure.json', 'r') as f:
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

# Load the tokenizer for BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess the dataset
def preprocess_function(examples):
    inputs = examples["input_text"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = examples["label"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="/home/xyro/Desktop/resultbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Ensure checkpoints are saved at each epoch
    save_total_limit=3,     # Limit the total number of checkpoints
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Function to train the model
def train_model(iteration):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(unique_labels))

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
    )

    # Override the _save method to handle tensor sharing issue
    def _save(self, output_dir: str):
        self.model.save_pretrained(output_dir, safe_serialization=False)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    trainer._save = types.MethodType(_save, trainer)

    # Train the model
    trainer.train()

# Loop to train the model multiple times
num_repeats = 10  # Set the number of times you want to repeat the training
for i in range(num_repeats):
    print(f"Training iteration {i+1}/{num_repeats}")
    train_model(i + 1)
    print(f"Completed training iteration {i+1}/{num_repeats}")

