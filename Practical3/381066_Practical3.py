# Install required libraries (run once)
# pip install transformers datasets torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

# 1. Create a small training dataset


stories = [
    "Once upon a time in a small village, a curious boy discovered a hidden door in an old tree.",
    "In a futuristic city where robots worked with humans, a young engineer created a machine that could read dreams.",
    "A lonely dragon living on a mountain found friendship with a brave little girl.",
    "Deep inside the ocean, a young mermaid discovered a glowing pearl with magical powers.",
    "In a magical forest, animals gathered every night to listen to stories told by an old wise owl."
]

dataset = Dataset.from_dict({"text": stories})

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 does not have padding token
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize)

# 4. Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./story_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1
)

# 6. Train model

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# 7. Save model


model.save_pretrained("./story_model")
tokenizer.save_pretrained("./story_model")

prompt = "Once upon a time in a magical forest"

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    inputs["input_ids"],
    max_length=120,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

story = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Story:\n")
print(story)