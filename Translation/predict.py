from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

def load_model_and_tokenizer(model_dir="./mbart-finetuned-en-te"):
    
    print("Loading the fine-tuned model and tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    tokenizer.src_lang = "en_XX"  
    return model, tokenizer

def translate_text(model, tokenizer, text, target_lang="te_IN", max_length=128):
    
    print(f"Translating text: {text}")
    tokenizer.src_lang = "en_XX" 
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation
  
model_dir = "./mbart-finetuned-en-te"  
model, tokenizer = load_model_and_tokenizer(model_dir)

english_text = "Newton's third law states that every action has an equal and opposite reaction."
telugu_translation = translate_text(model, tokenizer, english_text)
print("Translated Text:", telugu_translation)
