import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download("punkt_tab")

def load_models():
    tokenizer_for_qa = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    question_gen_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    return tokenizer_for_qa, qa_model, question_gen_pipeline, embedding_model

qa_tokenizer, qa_model, qg_pipeline, sentence_embedder = load_models()

def create_flashcards(text_input, max_flashcards):
    text_sentences = sent_tokenize(text_input)
    candidate_questions = []
    for sent in text_sentences:
        prompt = f"generate question: {sent.strip()}"
        try:
            question_output = qg_pipeline(prompt)
            candidate_questions.append(question_output[0]["generated_text"])
        except Exception:
            continue
    if not candidate_questions:
        raise ValueError("Unable to generate any questions. Please review the input text.")

    max_flashcards = min(len(candidate_questions), max_flashcards)

    question_vectors = sentence_embedder.encode(candidate_questions)
    avg_similarities = question_vectors.dot(question_vectors.T).mean(axis=1)
    best_indices = avg_similarities.argsort()[-max_flashcards:][::-1]
    selected_questions = [candidate_questions[i] for i in best_indices]

    extracted_answers = []
    for query in selected_questions:
        tokenized_pair = qa_tokenizer.encode_plus(query, text_input, add_special_tokens=True, return_tensors="pt")
        model_output = qa_model(**tokenized_pair)
        start_pos = torch.argmax(model_output.start_logits)
        end_pos = torch.argmax(model_output.end_logits) + 1
        response = qa_tokenizer.decode(tokenized_pair["input_ids"][0][start_pos:end_pos], skip_special_tokens=True)
        extracted_answers.append(response.strip())

    flashcard_deck = [{"Question": q, "Answer": a} for q, a in zip(selected_questions, extracted_answers)]
    return flashcard_deck, len(candidate_questions)

try:
    user_text = input("Please enter the text you want to generate flashcards from:\n")

    flashcards_output, possible_flashcards = create_flashcards(user_text, 5)
    print(f"The model can generate a maximum of {possible_flashcards} flashcards from your input text.")

    num_flashcards = int(input(f"How many flashcards would you like to generate? (Choose a number <= {possible_flashcards}): "))

    if num_flashcards > possible_flashcards:
        print(f"Error: The number of flashcards cannot exceed {possible_flashcards}. Using the maximum limit.")
        num_flashcards = possible_flashcards
    flashcards_output, _ = create_flashcards(user_text, num_flashcards)

    for idx, flashcard in enumerate(flashcards_output, start=1):
        print(f"Flashcard {idx}:")
        print(f"Q: {flashcard['Question']}")
        print(f"A: {flashcard['Answer']}")
        print("-" * 60)

except Exception as err:
    print(f"Error: {err}")
