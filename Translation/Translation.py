from datasets import load_dataset
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os

# Load a small subset of the dataset
print("Loading the dataset...")
dataset = load_dataset("Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized")
small_dataset = dataset["train"].shuffle(seed=42).select(range(2000))  # Use 2000 samples for quick fine-tuning
# Split the dataset into train and validation
print("Splitting dataset into train and validation sets...")
train_dataset = small_dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = small_dataset.train_test_split(test_size=0.1)["test"]

# Load mBART model and tokenizer
print("Loading mBART tokenizer and model...")
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Specify source and target languages
tokenizer.src_lang = "en_XX"
target_lang = "te_IN"

# Preprocess the dataset
def preprocess_data(examples):
    inputs = tokenizer(examples["output"], max_length=128, truncation=True, padding="max_length")
    targets = tokenizer(examples["telugu_output"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Tokenizing dataset...")
tokenized_train = train_dataset.map(preprocess_data, batched=True)
tokenized_eval = eval_dataset.map(preprocess_data, batched=True)

# Define training arguments
print("Setting up training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart-finetuned-en-te",
    evaluation_strategy="steps",  # Evaluate periodically
    save_strategy="steps",  # Save checkpoints periodically
    save_steps=10,  # Save every 10 steps
    eval_steps=10,  # Evaluate every 10 steps
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,  # One epoch for quick fine-tuning
    predict_with_generate=True,
    fp16=False,  # Disable fp16 for MPS
    bf16=False,
    no_cuda=True,  # Use CPU/MPS
    logging_dir="./logs",
    report_to="none",
    save_total_limit=1,  # Keep only the latest checkpoint
)

# Initialize the trainer
print("Initializing the trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,  # Add evaluation dataset
    tokenizer=tokenizer,
)

# Check for existing checkpoints
checkpoint_dir = "./mbart-finetuned-en-te/checkpoint-last"
resume_checkpoint = checkpoint_dir if os.path.isdir(checkpoint_dir) else None

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train(resume_from_checkpoint=resume_checkpoint)

# Save the fine-tuned model
print("Saving the fine-tuned model...")
trainer.save_model("./mbart-finetuned-en-te")

print("Fine-tuning complete!")
