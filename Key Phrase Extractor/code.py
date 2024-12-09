import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from torch.nn import CrossEntropyLoss

# Load the Dataset
dataset = load_dataset("midas/semeval2010") # Define Tokenizer and Model
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: O, B, I

# Label Mapping
label2id = {"O": 0, "B": 1, "I": 2}
id2label = {v: k for k, v in label2id.items()}
# Preprocessing Function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["document"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True,
    )
    labels = []
    for i, doc_tags in enumerate(examples["doc_bio_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignore special tokens
            else:
                label_ids.append(label2id.get(doc_tags[word_id], -100))
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
# Tokenize the Dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
# Compute Class Weights
class_distribution = Counter(
    label for sample in tokenized_dataset["train"]["labels"] for label in sample if label != -100
)
labels = list(class_distribution.keys())
counts = list(class_distribution.values())
class_weights = compute_class_weight(class_weight="balanced", classes=labels, y=[label for label, count in zip(labels, counts) for _ in range(count)])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
# Define WeightedTrainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(model.device)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, model.config.num_labels)
        labels = labels.view(-1)

        # Apply class weights
        loss_fn = CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,x
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    bf16=True,  # For MPS compatibility
    dataloader_num_workers=4,
)
# Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# Train the Model
trainer.train()

# Save the Model
model.save_pretrained("./fine_tuned_scibert")
tokenizer.save_pretrained("./fine_tuned_scibert")

# Evaluate the Model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

from transformers import pipeline

# Load fine-tuned SciBERT model
nlp = pipeline("token-classification", model="./fine_tuned_scibert", tokenizer="./fine_tuned_scibert")

def post_process_keyphrases(keyphrases):
    """
    Post-process extracted keyphrases to:
    1. Combine subwords (## tokens).
    2. Remove duplicates.
    3. Strip unwanted characters like punctuation and parentheses.
    4. Remove empty strings or isolated characters.
    """
    combined_phrases = []
    temp_phrase = ""

    for word in keyphrases:
        if word.startswith("##"):
            # Combine subword fragments
            temp_phrase += word[2:]
        else:
            if temp_phrase:
                combined_phrases.append(temp_phrase)
                temp_phrase = ""
            combined_phrases.append(word)

    if temp_phrase:
        combined_phrases.append(temp_phrase)

    # Remove unwanted characters and filter out empty or invalid phrases
    cleaned_phrases = [
        phrase.strip(".,()[]{}<>").lower()
        for phrase in combined_phrases
        if phrase.strip(".,()[]{}<>").strip()  # Remove empty or whitespace-only phrases
    ]

    # Remove duplicates
    cleaned_phrases = list(dict.fromkeys(cleaned_phrases))
    return cleaned_phrases

# Inference loop
while True:
    text = input("Enter text (or type 'exit' to quit): ")
    if text.lower() == "exit":
        break

    results = nlp(text)
    raw_keyphrases = [result["word"] for result in results if result["entity"] in ["LABEL_1", "LABEL_2"]]
    
    if raw_keyphrases:
        processed_keyphrases = post_process_keyphrases(raw_keyphrases)
        print("Extracted Keyphrases:", processed_keyphrases if processed_keyphrases else "No keyphrases found")
    else:
        print("Extracted Keyphrases: No keyphrases found")
