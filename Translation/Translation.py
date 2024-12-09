from datasets import load_dataset
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os

print("Loading the dataset...")
dataset = load_dataset("Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized")
small_dataset = dataset["train"].shuffle(seed=42).select(range(2000))  # Use 2000 samples for quick fine-tuning
print("Splitting dataset into train and validation sets...")
train_dataset = small_dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = small_dataset.train_test_split(test_size=0.1)["test"]

print("Loading mBART tokenizer and model...")
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

tokenizer.src_lang = "en_XX"
target_lang = "te_IN"

def preprocess_data(examples):
    inputs = tokenizer(examples["output"], max_length=128, truncation=True, padding="max_length")
    targets = tokenizer(examples["telugu_output"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Tokenizing dataset...")
tokenized_train = train_dataset.map(preprocess_data, batched=True)
tokenized_eval = eval_dataset.map(preprocess_data, batched=True)

print("Setting up training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart-finetuned-en-te",
    evaluation_strategy="steps", 
    save_strategy="steps", 
    save_steps=10, 
    eval_steps=10,  
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1, 
    predict_with_generate=True,
    fp16=False, 
    bf16=False,
    no_cuda=True, 
    logging_dir="./logs",
    report_to="none",
    save_total_limit=1,  
)

print("Initializing the trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval, 
    tokenizer=tokenizer,
)

checkpoint_dir = "./mbart-finetuned-en-te/checkpoint-last"
resume_checkpoint = checkpoint_dir if os.path.isdir(checkpoint_dir) else None

print("Starting fine-tuning...")
trainer.train(resume_from_checkpoint=resume_checkpoint)

print("Saving the fine-tuned model...")
trainer.save_model("./mbart-finetuned-en-te")

print("Fine-tuning complete!")
