from datasets import load_dataset
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")

dataset = load_dataset("Rogendo/English-Swahili-Sentence-Pairs", split="train[:10000]")

filtered_dataset = dataset.filter(lambda x: x["English sentence"] is not None)

def get_synonyms(word):
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def create_training_data(batch):
    masked_sentences = []
    target_words = []
    synonyms_list = []

    for sentence in batch["English sentence"]:
        words = sentence.split()
        for word in words:
            synonyms = get_synonyms(word.lower())
            if synonyms:
                masked_sentence = sentence.replace(word, "[MASK]")
                masked_sentences.append(masked_sentence)
                target_words.append(word)
                synonyms_list.append(synonyms)

    return {
        "masked_sentence": masked_sentences,
        "target_word": target_words,
        "synonyms": synonyms_list,
    }

preprocessed_data = filtered_dataset.map(
    create_training_data,
    batched=True,
    remove_columns=filtered_dataset.column_names,
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["masked_sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized_dataset = preprocessed_data.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15, 
)

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./bert-synonyms",
    evaluation_strategy="no",
    learning_rate=5e-5,
    num_train_epochs=1, 
    per_device_train_batch_size=8, 
    save_steps=500,
    save_total_limit=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  
)

trainer.train()

model.save_pretrained("./bert-synonyms")
tokenizer.save_pretrained("./bert-synonyms")

print("Model training completed and saved!")

def predict_synonyms(masked_sentence, top_k=5):
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    top_k_token_ids = logits[0, mask_token_index].squeeze().topk(top_k).indices.tolist()
    top_k_tokens = [tokenizer.decode(token_id).strip() for token_id in top_k_token_ids]
    
    return top_k_tokens

masked_sentence = "I enjoy eating [MASK] every day."
predicted_synonyms = predict_synonyms(masked_sentence, top_k=5)
print(f"Predicted synonyms: {predicted_synonyms}")
